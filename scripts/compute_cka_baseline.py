"""
Structural similarity baseline: computes Linear CKA between two models'
activations, with no stitch-layer training required. Use this to check
whether a pattern seen in run_cross_model_check.py (functional/stitching
similarity) is genuine structure or an artifact of the stitching training
process itself.

Usage:
    python scripts/compute_cka_baseline.py \
        --source_model wav2vec2-base --target_model hubert-base \
        --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked
from stitching import TemporalResample
from structural_similarity import linear_cka, flatten_for_cka
from sanity_checks import similarity_matrix, cross_model_correspondence_check, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_model", default="wav2vec2-base",
                     help="Any key registered in audio_models.py -- no choices= restriction, "
                          "load_model() validates and lists valid options if the key is wrong.")
    ap.add_argument("--target_model", default="hubert-base",
                     help="Any key registered in audio_models.py.")
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--extract_batch_size", type=int, default=8)
    ap.add_argument("--max_audio_samples", type=int, default=160000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading source model: {args.source_model} on {args.device}")
    source = load_model(args.source_model, device=args.device)
    print(f"Loading target model: {args.target_model} on {args.device}")
    target = load_model(args.target_model, device=args.device)

    if source.num_layers() != target.num_layers():
        print(f"ERROR: source has {source.num_layers()} layers, target has "
              f"{target.num_layers()} layers. This script assumes matching layer counts.")
        sys.exit(1)
    n_layers = list(range(1, source.num_layers() + 1))

    print(f"Loading {args.num_examples} audio examples (shared across both models)...")
    waveforms = get_audio_batch(args.num_examples, max_samples=args.max_audio_samples)

    print("Extracting hidden states for the SOURCE model...")
    source_input = source.preprocess(waveforms)
    source_hidden = extract_hidden_states_chunked(source, source_input, chunk_size=args.extract_batch_size)

    print("Extracting hidden states for the TARGET model...")
    target_input = target.preprocess(waveforms)
    target_hidden = extract_hidden_states_chunked(target, target_input, chunk_size=args.extract_batch_size)

    src_T = source_hidden[1].shape[1]
    tgt_T = target_hidden[1].shape[1]
    if src_T != tgt_T:
        print(f"NOTE: source T={src_T} != target T={tgt_T}; resampling source's time axis.")
        resampler = TemporalResample(target_frames=tgt_T)
        source_hidden = tuple(resampler(h) for h in source_hidden)
    else:
        print(f"Source and target frame counts match (T={src_T}); no resampling needed.")

    print("Computing Linear CKA for every layer pair (no training -- pure linear algebra)...")
    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    for i in n_layers:
        A = flatten_for_cka(source_hidden[i])
        for j in n_layers:
            B = flatten_for_cka(target_hidden[j])
            cka = linear_cka(A, B)
            scores[(n_layers.index(i), n_layers.index(j))] = cka
            done += 1
            print(f"\r  [{done}/{total_pairs}] {args.source_model} layer {i} -> "
                  f"{args.target_model} layer {j}: CKA={cka:.4f}", end="")
    print()

    sim = similarity_matrix(scores, len(n_layers), len(n_layers))
    result = cross_model_correspondence_check(sim)
    print_report(f"CKA Structural Baseline ({args.source_model} -> {args.target_model})",
                 result, layer_names=n_layers)

    print("\nFull CKA matrix (rows=source layer, cols=target layer):")
    header = "        " + "".join(f"{j:>8}" for j in n_layers)
    print(header)
    for i in n_layers:
        row = "".join(f"{sim[n_layers.index(i), n_layers.index(j)]:>8.4f}" for j in n_layers)
        print(f"layer{i:>3} {row}")

    print("\nCompare this matrix's argmax-per-row pattern against the stitching-based")
    print("cross_model_check result. If CKA also shows many source layers pointing at")
    print("the same target layer, that supports a genuine structural finding (e.g.")
    print("representational collapse in the target's later layers) rather than an")
    print("artifact of the stitching/training process.")


if __name__ == "__main__":
    main()
