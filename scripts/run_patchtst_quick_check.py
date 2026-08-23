"""
Phase D, Step 1: PatchTST -- combined structure verification + a small
self-stitch check, in ONE script/job (not a separate diagnose-then-run
cycle), deliberately to minimize total experiments given a time
constraint. Uses SYNTHETIC time-series data (noisy sine waves), not a
sourced real dataset -- adequate for testing representational
self-consistency, not forecasting quality. See PatchTSTStyleWrapper's
docstring in src/audio_models.py for full scope notes.

Usage:
    python scripts/run_patchtst_quick_check.py --num_examples 32
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from audio_models import load_model
from stitching import StitchLayer, lfm_loss, load_direct_matching_init, fraction_variance_explained
from sanity_checks import similarity_matrix, self_stitch_check, print_report


def make_synthetic_series(num_examples: int, length: int, seed: int = 0):
    """Noisy sine waves at varying frequencies/phases -- simple, fast,
    enough structure for a representational self-consistency check."""
    rng = np.random.RandomState(seed)
    series = []
    for _ in range(num_examples):
        freq = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        t = np.arange(length)
        s = amp * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, 0.1, size=length)
        series.append(s.astype(np.float32))
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=32)
    ap.add_argument("--max_layers_tested", type=int, default=6,
                     help="Cap on how many layers to test, evenly spaced, if the "
                          "model has more than this many -- keeps this quick.")
    ap.add_argument("--lfm_epochs", type=int, default=100)
    ap.add_argument("--lfm_lr", type=float, default=1e-3)
    ap.add_argument("--lfm_patience", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading patchtst on {args.device}...")
    model = load_model("patchtst", device=args.device)

    print("\n=== Structure verification (combined with the run, not a separate job) ===")
    print(f"context_length={model.context_length}, num_input_channels={model.num_input_channels}")
    print(f"num_layers() reports: {model.num_layers()}")

    print(f"\nGenerating {args.num_examples} synthetic time series (noisy sine waves)...")
    series = make_synthetic_series(args.num_examples, model.context_length)
    past_values = model.preprocess(series)
    print(f"preprocess() output shape: {tuple(past_values.shape)}")

    print("Running all_hidden_states()...")
    hidden_states = model.all_hidden_states(past_values)
    print(f"Number of hidden states: {len(hidden_states)} (expected: num_layers() = {model.num_layers()})")
    for i, h in enumerate(hidden_states):
        print(f"  hidden_states[{i}] shape: {tuple(h.shape)}")

    if len(hidden_states) != model.num_layers():
        print("\nMISMATCH -- hidden state count doesn't match num_layers().")
        print("Stopping here rather than proceeding with a possibly-wrong indexing assumption.")
        sys.exit(1)
    print("MATCH -- indexing convention holds. Proceeding to the self-stitch check.\n")
    print("NOTE: unlike every other model wrapped in this project, PatchTST's")
    print("hidden_states has NO separate pre-transformer embedding state -- layer i")
    print("(1-indexed, for display) maps to hidden_states[i-1], not hidden_states[i].\n")

    n_layers_total = model.num_layers()
    if n_layers_total <= args.max_layers_tested:
        n_layers = list(range(1, n_layers_total + 1))
    else:
        idx = np.linspace(1, n_layers_total, args.max_layers_tested, dtype=int)
        n_layers = sorted(set(idx.tolist()))
    print(f"Testing {len(n_layers)} of {n_layers_total} layers: {n_layers}\n")

    hidden_states = [h.to(args.device) for h in hidden_states]

    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    start_time = time.time()
    for i in n_layers:
        for j in n_layers:
            # i, j are 1-indexed layer NAMES for display; hidden_states is
            # 0-indexed with no separate embedding state, so shift by -1.
            src_feat_cpu = hidden_states[i - 1].cpu()
            tgt_feat_cpu = hidden_states[j - 1].cpu()
            src_feat = hidden_states[i - 1]
            tgt_feat = hidden_states[j - 1]

            stitch = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind="linear").to(args.device)
            load_direct_matching_init(stitch, src_feat_cpu, tgt_feat_cpu)

            opt = torch.optim.AdamW(stitch.parameters(), lr=args.lfm_lr)
            best_loss, patience_counter = float("inf"), 0
            for _ in range(args.lfm_epochs):
                opt.zero_grad()
                loss = lfm_loss(stitch(src_feat), tgt_feat)
                loss.backward()
                opt.step()
                lv = loss.item()
                if lv < best_loss - 1e-8:
                    best_loss, patience_counter = lv, 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.lfm_patience:
                        break

            with torch.no_grad():
                final_loss = lfm_loss(stitch(src_feat), tgt_feat).item()
            scores[(n_layers.index(i), n_layers.index(j))] = fraction_variance_explained(final_loss, tgt_feat)

            done += 1
            elapsed = time.time() - start_time
            print(f"\r  [{done}/{total_pairs}] layer {i} -> layer {j}: loss={final_loss:.6f} "
                  f"(elapsed {elapsed/60:.1f}m)", end="")
    print()

    sim = similarity_matrix(scores, len(n_layers), len(n_layers))
    result = self_stitch_check(sim)
    print_report("Self-Stitch Sanity Check (patchtst, linear stitcher, synthetic data, quick pass)",
                 result, layer_names=n_layers)

    print("\nNOTE: this used synthetic sine-wave data and a subset of layers, deliberately")
    print("scoped down for a fast first pass. Treat this as a smoke test of the pipeline")
    print("integration, not a fully-scaled Phase D result -- scale up (real time-series")
    print("data, all layers) once this is confirmed working.")


if __name__ == "__main__":
    main()
