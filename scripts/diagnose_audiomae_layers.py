"""
Diagnostic for the audiomae self-stitch near-ties (layers 4-8 all prefer
layer 3; layers 10,11 prefer their immediate neighbor). Tests whether this
is consistent with smooth, incremental residual-stream evolution (a known
property of pre-norm ViT blocks -- each block makes a small refinement
rather than a dramatic transformation, so nearby layers can be genuinely
hard for a LINEAR stitcher to tell apart) versus something else.

Computes, across all 12 layers:
  1. Per-layer activation variance -- context for scale differences.
  2. Adjacent-layer cosine similarity (layer i vs layer i+1) -- if this is
     high and roughly FLAT across the whole depth, that's consistent with
     smooth incremental evolution throughout the network, not something
     specific to layers 4-8 or 10-11.
  3. Every layer's cosine similarity to layer 3 specifically -- tests
     whether layer 3 is acting as a genuine "plateau center" that layers
     4-8 cluster around, versus layer 3 just being an arbitrary point on
     a smooth gradient (in which case similarity to layer 3 should DECAY
     smoothly with distance, not plateau).

Usage:
    python scripts/diagnose_audiomae_layers.py --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked


def cos_sim(a, b):
    """Mean cosine similarity over batch and token dims. a, b: (B, T, C)."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--extract_batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading audiomae on {args.device}...")
    model = load_model("audiomae", device=args.device)
    n_layers = model.num_layers()

    print(f"Loading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples, max_samples=160000)
    input_features = model.preprocess(waveforms)

    print("Extracting all hidden states...")
    hidden_states = extract_hidden_states_chunked(model, input_features, chunk_size=args.extract_batch_size)
    hidden_states = [h.to(args.device) for h in hidden_states]

    print(f"\n=== 1. Per-layer activation variance (layers 1-{n_layers}) ===")
    for i in range(1, n_layers + 1):
        var = hidden_states[i].var().item()
        print(f"  layer {i:>2}: var={var:.6f}")

    print(f"\n=== 2. Adjacent-layer cosine similarity (layer i vs layer i+1) ===")
    print("    High AND roughly flat across the whole depth -> consistent with")
    print("    smooth incremental residual evolution throughout the network.")
    adjacent_sims = []
    for i in range(1, n_layers):
        sim = cos_sim(hidden_states[i], hidden_states[i + 1])
        adjacent_sims.append(sim)
        marker = "  <-- one of the reported near-ties" if i in (3, 4, 5, 6, 7, 10, 11) else ""
        print(f"  layer {i:>2} vs layer {i+1:>2}: cos_sim={sim:.4f}{marker}")

    print(f"\n=== 3. Every layer's cosine similarity to layer 3 specifically ===")
    print("    Tests whether layer 3 is a genuine 'plateau center' for layers 4-8")
    print("    (similarity should stay HIGH and roughly FLAT across 4-8, not decay)")
    print("    versus just an arbitrary point on a smooth gradient (similarity")
    print("    should decay smoothly and monotonically with distance from layer 3).")
    layer3 = hidden_states[3]
    for i in range(1, n_layers + 1):
        sim = cos_sim(hidden_states[i], layer3)
        marker = "  <-- layer 3 itself" if i == 3 else ""
        print(f"  layer {i:>2} vs layer  3: cos_sim={sim:.4f}{marker}")

    print("\n=== Interpretation guide ===")
    print("  - If adjacent-layer similarity (#2) is high and FLAT across all 12 layers,")
    print("    not just around the failing pairs, the near-ties likely reflect a general")
    print("    property of this architecture's residual stream, not a localized issue.")
    print("  - If similarity-to-layer-3 (#3) stays high and flat across layers 4-8 specifically")
    print("    (rather than decaying smoothly), that supports a genuine 'plateau' -- several")
    print("    layers converging to a similar representation -- rather than simple smooth drift.")
    print("  - If similarity-to-layer-3 instead decays smoothly and monotonically with")
    print("    distance, layer 3 has no special status -- the near-ties are just an artifact")
    print("    of a smooth representational gradient, and a stricter accuracy threshold")
    print("    (or a more expressive stitcher) may be all that's needed.")


if __name__ == "__main__":
    main()
