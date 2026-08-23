"""
Final follow-up on the AudioMAE MLP-stitcher collapse: is layer 2
structurally simpler (lower effective rank / lower complexity) than the
other 11 layers? This would directly explain why it's a moderately easier
target to approximate from many different source layers (per the FVE +
OOD diagnostic run previously, which ruled out a TLT-style shortcut but
left the actual mechanism open).

For each of the 12 layers, computes:
  1. Activation variance -- already-familiar context.
  2. Effective rank via the participation ratio of the singular value
     spectrum: PR = (sum(sigma))^2 / sum(sigma^2), computed on the
     CENTERED activation matrix (like PCA). PR ranges from 1 (a single
     dominant direction -- maximally simple/low-rank) up to C=768 (a
     perfectly flat spectrum -- maximally complex/high-rank).
  3. The top-5 singular values, normalized by the largest -- a quick
     qualitative read on how sharply the spectrum decays.

Usage:
    python scripts/diagnose_audiomae_rank.py --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked


def participation_ratio(sigma: torch.Tensor) -> float:
    """PR = (sum(sigma))^2 / sum(sigma^2). Continuous effective-rank measure."""
    return (sigma.sum() ** 2 / (sigma ** 2).sum()).item()


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

    channel_dim = hidden_states[1].shape[-1]
    print(f"\nChannel dimension C={channel_dim} -- participation ratio ranges from 1 (simplest) to {channel_dim} (most complex).\n")

    print(f"{'Layer':>6} {'Variance':>12} {'Eff. Rank (PR)':>16}   Top-5 singular values (normalized)")
    results = []
    for i in range(1, n_layers + 1):
        feat = hidden_states[i].to(args.device)
        B, T, C = feat.shape
        flat = feat.reshape(B * T, C)
        centered = flat - flat.mean(dim=0, keepdim=True)

        var = feat.var().item()
        sigma = torch.linalg.svdvals(centered.float())
        pr = participation_ratio(sigma)
        top5 = (sigma[:5] / sigma[0]).tolist()

        marker = "  <-- collapse target" if i == 2 else ""
        top5_str = ", ".join(f"{v:.3f}" for v in top5)
        print(f"{i:>6} {var:>12.6f} {pr:>16.2f}   [{top5_str}]{marker}")
        results.append((i, var, pr))

    print("\nInterpretation guide:")
    print("  - A LOWER participation ratio (PR) means a MORE concentrated, lower-rank,")
    print("    structurally simpler representation -- a natural target for many different")
    print("    source layers to approximate well, regardless of true correspondence.")
    print("  - If layer 2's PR is noticeably lower than most other layers, that directly")
    print("    supports the hypothesis: layer 2 is structurally simple, which explains")
    print("    both signals from the prior diagnostic (easy FVE win, only mild/inconsistent")
    print("    OOD cost -- a simple target is easier to hit AND doesn't require going far")
    print("    out of distribution to approximate).")
    print("  - If layer 2's PR is unremarkable relative to the other layers, the")
    print("    reconstructability finding remains real but its cause is still open.")

    ranked = sorted(results, key=lambda r: r[2])
    print(f"\nLayers ranked by effective rank (lowest/simplest first):")
    for i, var, pr in ranked:
        marker = "  <-- collapse target" if i == 2 else ""
        print(f"  layer {i:>2}: PR={pr:.2f}{marker}")


if __name__ == "__main__":
    main()
