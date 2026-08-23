"""
Phase D, Step 2: PatchTST self-stitch check on REAL data -- the ETTh1
Electrical Transformer dataset, the SAME dataset this checkpoint
(namctin/patchtst_etth1_pretrain) was originally trained on. This
directly resolves two simplifications flagged in the Step 1 (synthetic)
report:
  - Real time-series data instead of synthetic sine waves.
  - Genuinely different signals per channel (the actual 7 ETTh1 columns:
    HUFL, HULL, MUFL, MULL, LUFL, LULL, OT), instead of one series
    replicated identically across all channels.

Data source: a plain public CSV on GitHub (the same URL used in HF's own
"getting started" PatchTST tutorial) -- no dataset library or auth needed.

Per-channel standardization (mean 0, std 1) is applied before feeding the
model, matching the original training pipeline's "scaling": "std" setup
(per the tutorial) -- skipping this would feed the model out-of-distribution
input scale and likely produce a misleading result.

Usage:
    python scripts/run_patchtst_real_data_check.py --num_examples 32
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

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_etth1_windows(num_examples: int, context_length: int, seed: int = 0):
    """Downloads ETTh1 (plain public CSV, no auth needed), extracts the 7
    real channels, standardizes each channel (mean 0, std 1) to match the
    original training pipeline's scaling, and builds `num_examples`
    windows of length `context_length` at random offsets."""
    import pandas as pd

    print(f"Downloading ETTh1 from {ETTH1_URL}...")
    df = pd.read_csv(ETTH1_URL)
    data = df[ETTH1_COLUMNS].values.astype(np.float32)  # (T_total, 7)
    print(f"  Loaded {data.shape[0]} timesteps x {data.shape[1]} channels.")

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    data = (data - mean) / std
    print(f"  Standardized per-channel (mean 0, std 1).")

    rng = np.random.RandomState(seed)
    max_start = data.shape[0] - context_length
    if max_start <= 0:
        raise ValueError(f"context_length ({context_length}) exceeds dataset length ({data.shape[0]}).")
    starts = rng.choice(max_start, size=min(num_examples, max_start), replace=False)

    windows = np.stack([data[s:s + context_length] for s in starts])  # (B, context_length, 7)
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=32)
    ap.add_argument("--lfm_epochs", type=int, default=100)
    ap.add_argument("--lfm_lr", type=float, default=1e-3)
    ap.add_argument("--lfm_patience", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading patchtst on {args.device}...")
    model = load_model("patchtst", device=args.device)
    print(f"context_length={model.context_length}, num_input_channels={model.num_input_channels}, "
          f"num_layers={model.num_layers()}")
    channel_attention = getattr(model.model.config, "channel_attention", None)
    print(f"channel_attention={channel_attention}")
    if not channel_attention:
        print("NOTE: channel_attention is OFF -- channels do not attend to each other, so")
        print("channel 0's computed representation depends ONLY on channel 0's own real")
        print("values, not on what the other 6 channels contain. This test's real")
        print("improvement over the synthetic version is REAL SIGNAL STATISTICS for the")
        print("analyzed channel, not multi-channel interaction -- stated precisely so this")
        print("isn't overclaimed.")

    windows = load_etth1_windows(args.num_examples, model.context_length)
    print(f"Built {windows.shape[0]} real windows, shape {windows.shape}.")
    past_values = torch.from_numpy(windows).float().to(args.device)

    print("Running all_hidden_states() on REAL data...")
    hidden_states = model.all_hidden_states(past_values)
    print(f"Number of hidden states: {len(hidden_states)} (expected: num_layers() = {model.num_layers()})")
    for i, h in enumerate(hidden_states):
        print(f"  hidden_states[{i}] shape: {tuple(h.shape)}")

    if len(hidden_states) != model.num_layers():
        print("\nMISMATCH -- stopping rather than proceeding with a wrong indexing assumption.")
        sys.exit(1)
    print("MATCH -- proceeding to the self-stitch check.\n")

    n_layers = list(range(1, model.num_layers() + 1))
    hidden_states = [h.to(args.device) for h in hidden_states]

    scores = {}
    total_pairs = len(n_layers) * len(n_layers)
    done = 0
    start_time = time.time()
    for i in n_layers:
        for j in n_layers:
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
    print_report("Self-Stitch Sanity Check (patchtst, linear stitcher, REAL ETTh1 data)",
                 result, layer_names=n_layers)

    print("\nThis used REAL data (ETTh1, the model's own pretraining dataset) with genuinely")
    print("different signals across all 7 channels -- a real test, not a synthetic smoke test.")


if __name__ == "__main__":
    main()
