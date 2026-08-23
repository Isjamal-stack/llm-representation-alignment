"""
Diagnostic for ClapStyleWrapper -- run this BEFORE the real self-stitch
sweep, same discipline as the Whisper signature check.

Usage:
    python scripts/diagnose_clap_shapes.py --num_examples 8
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
    ap.add_argument("--num_examples", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading clap-htsat-unfused on {args.device}...")
    model = load_model("clap-htsat-unfused", device=args.device)
    print(f"model.num_layers() reports: {model.num_layers()}")
    print(f"Raw config.depths: {model.model.config.depths}")

    print(f"\nLoading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples, max_samples=160000)
    input_features = model.preprocess(waveforms)
    print(f"preprocess() output shape: {tuple(input_features.shape)}")

    with torch.no_grad():
        raw_out = model.model(input_features, output_hidden_states=True)
    print(f"\nNumber of raw hidden_states returned: {len(raw_out.hidden_states)}")
    for i, h in enumerate(raw_out.hidden_states):
        print(f"  hidden_states[{i}] raw shape: {tuple(h.shape)}")

    print("\nNow via our wrapper's all_hidden_states() (flattened to (B, T, C)):")
    flattened = model.all_hidden_states(input_features)
    print(f"Number of flattened hidden states: {len(flattened)}")
    for i, h in enumerate(flattened):
        print(f"  flattened[{i}] shape: {tuple(h.shape)}")

    print("\nSanity check: does len(flattened) == num_layers() + 1?")
    print(f"  len(flattened) = {len(flattened)}, num_layers()+1 = {model.num_layers() + 1}")
    if len(flattened) == model.num_layers() + 1:
        print("  MATCH -- run_phase_a_sanity.py's indexing convention will work correctly.")
    else:
        print("  MISMATCH -- report this back before running the real sweep.")


if __name__ == "__main__":
    main()
