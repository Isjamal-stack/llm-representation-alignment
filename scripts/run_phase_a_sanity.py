"""
Phase A, Step 1: pipeline validation via the self-stitch sanity check.

Usage:
    python scripts/run_phase_a_sanity.py --model wav2vec2-base --num_examples 64
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked
from stitching import (
    StitchLayer, TemporalResample, StitchedModel,
    lfm_loss, direct_matching_init, load_direct_matching_init, fraction_variance_explained,
)
from sanity_checks import similarity_matrix, self_stitch_check, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-base")
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Which layers to test; defaults to all encoder layers.")
    ap.add_argument("--stitch_kind", default="linear", choices=["linear", "mlp", "conv"])
    ap.add_argument("--lfm_epochs", type=int, default=200)
    ap.add_argument("--lfm_lr", type=float, default=1e-3)
    ap.add_argument("--lfm_patience", type=int, default=20,
                     help="Stop a pair's training early if loss hasn't improved for this many epochs.")
    ap.add_argument("--extract_batch_size", type=int, default=8,
                     help="Sub-batch size for the initial forward pass, to bound peak GPU memory.")
    ap.add_argument("--max_audio_samples", type=int, default=160000,
                     help="Truncate each clip to at most this many raw samples "
                          "(default 160000 = 10s at 16kHz) so one long outlier "
                          "clip can't blow up the whole batch's padded length.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading model: {args.model} on {args.device}")
    model = load_model(args.model, device=args.device)
    n_layers = args.layers or list(range(1, model.num_layers() + 1))

    print(f"Loading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples, max_samples=args.max_audio_samples)
    input_values = model.preprocess(waveforms)

    print(f"Extracting all hidden states once, in chunks of {args.extract_batch_size} "
          f"(cached on CPU, reused for every layer pair)...")
    hidden_states = extract_hidden_states_chunked(model, input_values, chunk_size=args.extract_batch_size)

    def align_T(src_feat_cpu, tgt_feat_cpu):
        src_T, tgt_T = src_feat_cpu.shape[1], tgt_feat_cpu.shape[1]
        if src_T == tgt_T:
            return src_feat_cpu
        resampler = TemporalResample(target_frames=tgt_T)
        return resampler(src_feat_cpu)

    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    start_time = time.time()
    for i in n_layers:
        for j in n_layers:
            src_feat_cpu = align_T(hidden_states[i], hidden_states[j])
            tgt_feat_cpu = hidden_states[j]
            src_feat = src_feat_cpu.to(args.device)
            tgt_feat = tgt_feat_cpu.to(args.device)

            stitch = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind=args.stitch_kind).to(args.device)

            if args.stitch_kind == "linear":
                load_direct_matching_init(stitch, src_feat_cpu, tgt_feat_cpu)

            opt = torch.optim.AdamW(stitch.parameters(), lr=args.lfm_lr)
            best_loss = float("inf")
            patience_counter = 0
            for _ in range(args.lfm_epochs):
                opt.zero_grad()
                pred = stitch(src_feat)
                loss = lfm_loss(pred, tgt_feat)
                loss.backward()
                opt.step()

                loss_val = loss.item()
                if loss_val < best_loss - 1e-8:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.lfm_patience:
                        break

            with torch.no_grad():
                final_loss = lfm_loss(stitch(src_feat), tgt_feat).item()
            scores[(n_layers.index(i), n_layers.index(j))] = fraction_variance_explained(final_loss, tgt_feat)

            del src_feat, tgt_feat, stitch, opt
            done += 1
            elapsed = time.time() - start_time
            eta_min = (elapsed / done) * (total_pairs - done) / 60
            print(f"\r  [{done}/{total_pairs}] layer {i} -> layer {j}: loss={final_loss:.6f}  "
                  f"(elapsed {elapsed/60:.1f}m, ETA {eta_min:.1f}m)", end="")
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
