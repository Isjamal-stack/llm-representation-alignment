"""
Head-to-head follow-up: layer 1 has an even LOWER effective rank than
layer 2 (PR=77.00 vs 86.50), yet layer 2 -- not layer 1 -- was the
collapse target in the original MLP sweep (layer 1 itself even preferred
layer 2 over its own match). If pure simplicity/rank were the whole
story, layer 1 should have been at least as attractive a target.

This directly pits layer 1 against layer 2 as competing targets for every
other layer, training BOTH i->1 and i->2 stitchers and comparing FVE
head-to-head -- not just which one an earlier sweep's argmax happened to
pick, but the actual margin between them. This distinguishes two
possibilities:
  - Layer 1 is a genuinely POOR target (loses by a wide, consistent
    margin) despite its low rank -- rank alone doesn't explain
    "attractiveness" as a stitch target; something else about layer 2
    specifically matters.
  - Layer 1 is actually NEARLY AS GOOD a target (close margin, sometimes
    even winning) -- the original "layer 2 specifically" finding may be a
    near-arbitrary tie-break between two similarly easy low-rank targets,
    not evidence layer 2 is uniquely special.

Usage:
    python scripts/diagnose_audiomae_layer1_vs_layer2.py --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked
from stitching import StitchLayer, lfm_loss, load_direct_matching_init, fraction_variance_explained

SOURCE_LAYERS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # all except layer 2 itself


def train_and_score(src_feat, tgt_feat, src_feat_cpu, tgt_feat_cpu, stitch_kind, epochs, lr, patience):
    stitch = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind=stitch_kind).to(src_feat.device)
    if stitch_kind == "linear":
        load_direct_matching_init(stitch, src_feat_cpu, tgt_feat_cpu)
    opt = torch.optim.AdamW(stitch.parameters(), lr=lr)
    best_loss, patience_counter = float("inf"), 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = lfm_loss(stitch(src_feat), tgt_feat)
        loss.backward()
        opt.step()
        lv = loss.item()
        if lv < best_loss - 1e-8:
            best_loss, patience_counter = lv, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    with torch.no_grad():
        final_loss = lfm_loss(stitch(src_feat), tgt_feat).item()
    return fraction_variance_explained(final_loss, tgt_feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--stitch_kind", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--lfm_epochs", type=int, default=200)
    ap.add_argument("--lfm_lr", type=float, default=1e-3)
    ap.add_argument("--lfm_patience", type=int, default=20)
    ap.add_argument("--extract_batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading audiomae on {args.device}...")
    model = load_model("audiomae", device=args.device)

    print(f"Loading {args.num_examples} audio examples...")
    waveforms = get_audio_batch(args.num_examples, max_samples=160000)
    input_features = model.preprocess(waveforms)

    print("Extracting all hidden states...")
    hidden_states = extract_hidden_states_chunked(model, input_features, chunk_size=args.extract_batch_size)

    target1_cpu, target1 = hidden_states[1], hidden_states[1].to(args.device)
    target2_cpu, target2 = hidden_states[2], hidden_states[2].to(args.device)

    print(f"\n{'Source':>6} {'FVE vs layer 1':>15} {'FVE vs layer 2':>15}   {'Winner':>10}   {'Margin':>8}")
    wins_1, wins_2 = 0, 0
    for i in SOURCE_LAYERS:
        src_cpu = hidden_states[i]
        src = src_cpu.to(args.device)

        fve_1 = train_and_score(src, target1, src_cpu, target1_cpu, args.stitch_kind,
                                 args.lfm_epochs, args.lfm_lr, args.lfm_patience)
        fve_2 = train_and_score(src, target2, src_cpu, target2_cpu, args.stitch_kind,
                                 args.lfm_epochs, args.lfm_lr, args.lfm_patience)

        winner = "layer 1" if fve_1 > fve_2 else "layer 2"
        margin = abs(fve_1 - fve_2)
        if fve_1 > fve_2:
            wins_1 += 1
        else:
            wins_2 += 1

        print(f"{i:>6} {fve_1:>15.4f} {fve_2:>15.4f}   {winner:>10}   {margin:>8.4f}")

    print(f"\nHead-to-head record: layer 1 wins {wins_1}/{len(SOURCE_LAYERS)}, "
          f"layer 2 wins {wins_2}/{len(SOURCE_LAYERS)}")

    print("\nInterpretation guide:")
    print("  - If layer 2 wins nearly every head-to-head matchup by a CONSISTENT,")
    print("    non-trivial margin, layer 2 is genuinely more attractive as a target")
    print("    than its rank alone would predict -- something else about it matters.")
    print("  - If the record is close (e.g. layer 1 wins a meaningful fraction, or")
    print("    margins are consistently tiny), the original finding was closer to an")
    print("    arbitrary tie-break between two similarly-easy low-rank targets --")
    print("    layer 2 isn't uniquely special, it just edged out layer 1 by chance")
    print("    in the specific sweep that was run.")


if __name__ == "__main__":
    main()
