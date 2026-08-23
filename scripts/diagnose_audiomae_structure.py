"""
Diagnostic for AudioMAEStyleWrapper -- run this BEFORE the real self-stitch
sweep, same discipline as the Whisper signature check and CLAP shape check.

Usage:
    python scripts/diagnose_audiomae_structure.py --num_examples 4
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading audiomae on {args.device}...")
    model = load_model("audiomae", device=args.device)

    print("\nChecking expected timm ViT attributes exist on the loaded model:")
    for attr in ["patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm"]:
        present = hasattr(model.model, attr)
        val = getattr(model.model, attr, None)
        print(f"  {attr}: present={present}"
              + (f", type={type(val).__name__}" if present else ""))

    print(f"\nnum_layers() reports: {model.num_layers()}")
    print(f"Number of blocks in model.model.blocks: {len(model.model.blocks)}")

    print(f"\nLoading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples, max_samples=160000)
    input_features = model.preprocess(waveforms)
    print(f"preprocess() output shape: {tuple(input_features.shape)}  "
          f"(expected: ({args.num_examples}, 1, 1024, 128))")

    print("\nRunning all_hidden_states()...")
    hidden_states = model.all_hidden_states(input_features)
    print(f"Number of hidden states: {len(hidden_states)}  "
          f"(expected: num_layers()+1 = {model.num_layers() + 1})")
    for i, h in enumerate(hidden_states):
        print(f"  hidden_states[{i}] shape: {tuple(h.shape)}")

    print("\nSanity check: does len(hidden_states) == num_layers() + 1?")
    if len(hidden_states) == model.num_layers() + 1:
        print("  MATCH -- run_phase_a_sanity.py's indexing convention will work correctly.")
    else:
        print("  MISMATCH -- report this back before running the real sweep.")

    print("\nSanity check: do all hidden states share the same shape "
          "(expected for a flat ViT, unlike CLAP's shrinking Swin stages)?")
    shapes = set(tuple(h.shape) for h in hidden_states)
    if len(shapes) == 1:
        print(f"  MATCH -- all {len(hidden_states)} hidden states are shape {shapes.pop()}. "
              f"No temporal resampling needed for AudioMAE (unlike CLAP).")
    else:
        print(f"  MISMATCH -- found {len(shapes)} distinct shapes: {shapes}. "
              f"This is unexpected for a flat ViT -- investigate before the real run.")


if __name__ == "__main__":
    main()
