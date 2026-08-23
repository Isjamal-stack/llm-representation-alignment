"""
Phase A, Step 2: cross-model correspondence check.

Where the self-stitch check (run_phase_a_sanity.py) asks "is a layer most
similar to itself, within one model?", this asks the next question up:
"across two DIFFERENT models, is the architecturally corresponding layer
still the most similar one?" This is the second mandatory sanity check
from the reliability paper, applied here for the first time in audio.

Scope note: the protocol's ideal version of this check uses two
INDEPENDENTLY-TRAINED SEEDS of the identical architecture (e.g. two
from-scratch wav2vec2-base pretraining runs with different random
initializations) -- that isolates "does correspondence survive
re-training" from "does correspondence survive an entirely different
model". We don't have two such seeds sitting around (that requires
pretraining a model from scratch, well beyond Phase A's scope), so this
script instead uses two DIFFERENT pretrained checkpoints that share the
same architecture family and layer count -- wav2vec2-base and
hubert-base by default. This is a real, meaningful check (do
same-shaped models trained on different objectives still line up
layer-for-layer?), just a different -- and for our purposes, more
practically available -- question than the strict same-seed version.
This distinction should be made explicit in any writeup of results.

Usage:
    python scripts/run_cross_model_check.py \
        --source_model wav2vec2-base --target_model hubert-base \
        --num_examples 64
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
    StitchLayer, TemporalResample,
    lfm_loss, load_direct_matching_init, fraction_variance_explained,
)
from sanity_checks import similarity_matrix, cross_model_correspondence_check, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_model", default="wav2vec2-base",
                     help="Any key registered in audio_models.py -- no choices= restriction, "
                          "load_model() validates and lists valid options if the key is wrong.")
    ap.add_argument("--target_model", default="hubert-base",
                     help="Any key registered in audio_models.py.")
    ap.add_argument("--num_examples", type=int, default=64)
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

    if args.source_model == args.target_model:
        print("NOTE: source and target are the same model -- this degenerates to the "
              "self-stitch check. Use run_phase_a_sanity.py for that instead.")

    print(f"Loading source model: {args.source_model} on {args.device}")
    source = load_model(args.source_model, device=args.device)
    print(f"Loading target model: {args.target_model} on {args.device}")
    target = load_model(args.target_model, device=args.device)

    if source.num_layers() != target.num_layers():
        print(f"ERROR: source has {source.num_layers()} layers, target has "
              f"{target.num_layers()} layers. This script currently assumes matching "
              f"layer counts (position i in source corresponds to position i in target). "
              f"Mismatched depths aren't supported yet -- see cross_model_correspondence_check "
              f"in src/sanity_checks.py if you need to extend this.")
        sys.exit(1)
    n_layers = list(range(1, source.num_layers() + 1))

    print(f"Loading {args.num_examples} audio examples (shared across both models)...")
    waveforms = get_audio_batch(args.num_examples, max_samples=args.max_audio_samples)

    print("Preprocessing and extracting hidden states for the SOURCE model...")
    source_input = source.preprocess(waveforms)
    source_hidden = extract_hidden_states_chunked(source, source_input, chunk_size=args.extract_batch_size)

    print("Preprocessing and extracting hidden states for the TARGET model...")
    target_input = target.preprocess(waveforms)
    target_hidden = extract_hidden_states_chunked(target, target_input, chunk_size=args.extract_batch_size)

    # Same raw waveforms + same downsampling ratio across wav2vec2-family
    # models means source and target should already agree on T (the frame
    # count after the conv stack). We check rather than assume, and fall
    # back to temporal resampling if they don't match -- this is the first
    # time TemporalResample actually gets exercised in this project.
    src_T = source_hidden[1].shape[1]
    tgt_T = target_hidden[1].shape[1]
    if src_T != tgt_T:
        print(f"NOTE: source T={src_T} != target T={tgt_T}; resampling source's "
              f"time axis to match target before stitching.")
        resampler = TemporalResample(target_frames=tgt_T)
        source_hidden = tuple(resampler(h) for h in source_hidden)
    else:
        print(f"Source and target frame counts match (T={src_T}); no temporal resampling needed.")

    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    start_time = time.time()
    for i in n_layers:
        src_feat_cpu = source_hidden[i]
        for j in n_layers:
            tgt_feat_cpu = target_hidden[j]
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
            print(f"\r  [{done}/{total_pairs}] {args.source_model} layer {i} -> "
                  f"{args.target_model} layer {j}: loss={final_loss:.6f}  "
                  f"(elapsed {elapsed/60:.1f}m, ETA {eta_min:.1f}m)", end="")
    print()

    sim = similarity_matrix(scores, len(n_layers), len(n_layers))
    result = cross_model_correspondence_check(sim)
    print_report(f"Cross-Model Correspondence Check "
                 f"({args.source_model} -> {args.target_model}, {args.stitch_kind} stitcher)",
                 result, layer_names=n_layers)

    if result["accuracy"] < 1.0:
        print("\n⚠️  Correspondence check failed for one or more layers.")
        print("    This can be a genuine finding (per Kornblith et al. 2019, even")
        print("    LCKA -- the best-performing structural index in vision -- doesn't")
        print("    always get 100% here). Compare against a structural similarity")
        print("    index (e.g. CKA) before concluding the stitching pipeline itself")
        print("    is at fault.")
    else:
        print("\n✅ Architecturally corresponding layers are correctly identified")
        print("   as most similar across the two models.")


if __name__ == "__main__":
    main()
