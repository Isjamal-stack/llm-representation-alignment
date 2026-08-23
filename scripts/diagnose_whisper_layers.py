"""
Diagnostic for the whisper-base layer 1 <-> layer 2 near-tie found in the
self-stitch check. Tests two hypotheses:

  1. Padding dominance: Whisper pads every clip to a FIXED 30 seconds.
     If real clips are much shorter, most of the 1500-frame input is
     silence padding -- early layers might process that dominant silent
     region near-identically, making them hard to tell apart for reasons
     that have nothing to do with the stitching pipeline.

  2. Genuine architectural redundancy: layers 1 and 2 might just produce
     very similar representations on real (non-padding) content too --
     which would make this a real finding about Whisper's encoder, not
     an artifact of the padding.

Run interactively (no batch job needed, this is quick):
    python scripts/diagnose_whisper_layers.py --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== Hypothesis 1: How much of the 30s window is real audio vs. padding? ===")
    waveforms = get_audio_batch(args.num_examples, max_samples=160000)  # cap matches rest of project
    durations = [len(w) / 16000 for w in waveforms]
    print(f"  Clip durations (s): min={min(durations):.2f}  "
          f"max={max(durations):.2f}  mean={sum(durations)/len(durations):.2f}")
    print(f"  Whisper's fixed window: 30.00s")
    print(f"  --> On average, {100 * (1 - (sum(durations)/len(durations))/30):.1f}% of "
          f"every input to whisper-base is silence padding, not real audio.")

    print("\n=== Hypothesis 2: How similar are layer 1 and layer 2's REAL activations? ===")
    model = load_model("whisper-base", device=args.device)
    input_features = model.preprocess(waveforms)
    hidden_states = extract_hidden_states_chunked(model, input_features, chunk_size=8)

    layer1 = hidden_states[1].to(args.device)  # (B, T, C)
    layer2 = hidden_states[2].to(args.device)

    # Cosine similarity per (example, frame), averaged -- computed separately
    # for the REAL-AUDIO region vs. the PADDING region, so we can tell the
    # two hypotheses apart directly rather than guessing.
    frame_rate = layer1.shape[1] / 30.0  # frames per second, given the fixed 30s window
    cos = torch.nn.functional.cosine_similarity(layer1, layer2, dim=-1)  # (B, T)

    real_sims, pad_sims = [], []
    for i, dur in enumerate(durations[:layer1.shape[0]]):
        real_frames = int(dur * frame_rate)
        real_frames = min(real_frames, layer1.shape[1])
        if real_frames > 0:
            real_sims.append(cos[i, :real_frames].mean().item())
        if real_frames < layer1.shape[1]:
            pad_sims.append(cos[i, real_frames:].mean().item())

    print(f"  Mean cosine similarity (layer1 vs layer2), REAL-AUDIO frames only: "
          f"{sum(real_sims)/len(real_sims):.4f}")
    if pad_sims:
        print(f"  Mean cosine similarity (layer1 vs layer2), PADDING frames only:    "
              f"{sum(pad_sims)/len(pad_sims):.4f}")
    else:
        print("  (No padding frames found -- all clips filled the full 30s window.)")

    print("\nInterpretation guide:")
    print("  - If PADDING similarity is much HIGHER than REAL-AUDIO similarity,")
    print("    that supports Hypothesis 1 (padding dominance drives the near-tie).")
    print("  - If REAL-AUDIO similarity is ALSO high (close to padding's), that")
    print("    supports Hypothesis 2 (layers 1-2 are genuinely near-redundant,")
    print("    a real property of whisper-base's encoder, not a padding artifact).")


if __name__ == "__main__":
    main()
