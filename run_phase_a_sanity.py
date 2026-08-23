"""
Phase A, Step 1: pipeline validation via the self-stitch sanity check.

This is the FIRST thing to run -- before any TLT/FFM sweep, before OOD
detectors, before touching a second model. It answers one question:
"does my stitching pipeline behave sanely at all?"

Usage:
    python scripts/run_phase_a_sanity.py --model wav2vec2-base --num_examples 64

Requires: a GPU machine with network access (to pull the checkpoint and a
small audio dataset), and the packages in requirements.txt installed.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import load_dataset
from audio_models import load_model
from stitching import (
    StitchLayer, TemporalResample, StitchedModel,
    lfm_loss, direct_matching_init, load_direct_matching_init,
)
from sanity_checks import similarity_matrix, self_stitch_check, print_report


def get_audio_batch(num_examples: int, sampling_rate: int = 16000):
    """Pulls a small slice of a public speech dataset. Swap this out for
    whatever dataset you land on for Phase A (e.g. a SUPERB task subset).

    NOTE: we deliberately disable the `datasets` library's automatic audio
    decoding (`Audio(decode=False)`) and decode manually with `soundfile`
    instead. Recent `datasets` versions (5.x) require an extra package,
    `torchcodec`, for their built-in decode path -- decoding manually avoids
    that dependency entirely and keeps this script working across versions.
    """
    import io
    import soundfile as sf
    from datasets import Audio

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.select(range(min(num_examples, len(ds))))
    ds = ds.cast_column("audio", Audio(decode=False))

    waveforms = []
    for ex in ds:
        audio = ex["audio"]
        if audio.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(audio["bytes"]))
        else:
            data, sr = sf.read(audio["path"])
        waveforms.append(data)
    return waveforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-base")
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Which layers to test; defaults to all encoder layers.")
    ap.add_argument("--stitch_kind", default="linear", choices=["linear", "mlp", "conv"])
    ap.add_argument("--lfm_epochs", type=int, default=200)
    ap.add_argument("--lfm_lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading model: {args.model} on {args.device}")
    model = load_model(args.model, device=args.device)
    n_layers = args.layers or list(range(1, model.num_layers() + 1))

    print(f"Loading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples)
    input_values = model.preprocess(waveforms)

    print("Extracting all hidden states once (cached, reused for every layer pair)...")
    with torch.no_grad():
        hidden_states = model.all_hidden_states(input_values)  # tuple, index 0..num_layers

    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    for i in n_layers:
        src_feat = hidden_states[i].detach()
        for j in n_layers:
            tgt_feat = hidden_states[j].detach()  # same T here (same model, no resampling needed)

            stitch = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind=args.stitch_kind).to(args.device)

            if args.stitch_kind == "linear":
                # Warm-start with the closed-form Direct Matching solution,
                # exactly as in the reference recipe.
                load_direct_matching_init(stitch, src_feat.cpu(), tgt_feat.cpu())

            opt = torch.optim.AdamW(stitch.parameters(), lr=args.lfm_lr)
            for _ in range(args.lfm_epochs):
                opt.zero_grad()
                pred = stitch(src_feat)
                loss = lfm_loss(pred, tgt_feat)
                loss.backward()
                opt.step()

            with torch.no_grad():
                final_loss = lfm_loss(stitch(src_feat), tgt_feat).item()
            # similarity = negative loss (lower loss = higher similarity)
            scores[(n_layers.index(i), n_layers.index(j))] = -final_loss

            done += 1
            print(f"\r  [{done}/{total_pairs}] layer {i} -> layer {j}: loss={final_loss:.6f}", end="")
    print()

    sim = similarity_matrix(scores, len(n_layers), len(n_layers))
    result = self_stitch_check(sim)
    print_report(f"Self-Stitch Sanity Check ({args.model}, {args.stitch_kind} stitcher)",
                 result, layer_names=n_layers)

    if result["accuracy"] < 1.0:
        print("\n⚠️  Sanity check failed for one or more layers.")
        print("    Before scaling up: check temporal alignment, learning rate,")
        print("    number of LFM epochs, and stitch layer capacity.")
    else:
        print("\n✅ Pipeline passes the self-stitch sanity check. Safe to proceed")
        print("   to cross-model stitching and the TLT/FFM comparison.")


if __name__ == "__main__":
    main()
