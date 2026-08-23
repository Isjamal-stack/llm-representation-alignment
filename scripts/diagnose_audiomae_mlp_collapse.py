"""
Diagnostic for the AudioMAE MLP-stitcher collapse onto layer 2. The MLP
self-stitch sweep showed layers 1,4-12 all preferring layer 2 over their
own true match -- a dramatic collapse, unlike the linear stitcher's smooth
near-ties. This tests whether that's a genuine shortcut (the MLP finds an
easy way to approximate layer 2's activations without real correspondence,
the same category of problem the reliability paper warns about for TLT)
or a real, if surprising, structural similarity.

For each layer i that wrongly preferred layer 2, trains TWO MLP stitchers:
  - i -> 2   (the "wrong" pair the sweep preferred)
  - i -> i   (the true, correct pair)
and reports BOTH the FVE score (matches the original sweep) AND the OOD
proxy distance (src/ood_detector.py: simple_distribution_distance) against
each target's REAL activations. If i->2 has similar or LOWER OOD distance
than i->i despite being architecturally "wrong", that's consistent with a
genuine (if surprising) similarity. If i->2 shows a MARKEDLY higher OOD
distance than i->i, that's a shortcut signature: the MLP is reconstructing
something that scores well on FVE without actually resembling layer 2's
real distribution.

Usage:
    python scripts/diagnose_audiomae_mlp_collapse.py --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from audio_models import load_model
from extraction_utils import get_audio_batch, extract_hidden_states_chunked
from stitching import StitchLayer, lfm_loss, load_direct_matching_init, fraction_variance_explained
from ood_detector import simple_distribution_distance

# The layers that wrongly preferred layer 2 in the real MLP sweep (n=64).
WRONG_LAYERS = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
COLLAPSE_TARGET = 2


def train_stitch(src_feat, tgt_feat, src_feat_cpu, tgt_feat_cpu, stitch_kind, epochs, lr, patience):
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
        stitched = stitch(src_feat)
    return stitched, final_loss


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

    target2_cpu = hidden_states[COLLAPSE_TARGET]
    target2 = target2_cpu.to(args.device)

    print(f"\n{'Layer':>6} {'WRONG (i->2) FVE':>18} {'WRONG OOD-dist':>16}   "
          f"{'TRUE (i->i) FVE':>18} {'TRUE OOD-dist':>15}")
    for i in WRONG_LAYERS:
        src_cpu = hidden_states[i]
        src = src_cpu.to(args.device)
        true_tgt_cpu = hidden_states[i]
        true_tgt = true_tgt_cpu.to(args.device)

        # WRONG pair: i -> 2
        stitched_wrong, loss_wrong = train_stitch(
            src, target2, src_cpu, target2_cpu, args.stitch_kind,
            args.lfm_epochs, args.lfm_lr, args.lfm_patience,
        )
        fve_wrong = fraction_variance_explained(loss_wrong, target2)
        ood_wrong = simple_distribution_distance(stitched_wrong, target2)

        # TRUE pair: i -> i
        stitched_true, loss_true = train_stitch(
            src, true_tgt, src_cpu, true_tgt_cpu, args.stitch_kind,
            args.lfm_epochs, args.lfm_lr, args.lfm_patience,
        )
        fve_true = fraction_variance_explained(loss_true, true_tgt)
        ood_true = simple_distribution_distance(stitched_true, true_tgt)

        print(f"{i:>6} {fve_wrong:>18.4f} {ood_wrong:>16.4f}   {fve_true:>18.4f} {ood_true:>15.4f}")

    print("\nInterpretation guide:")
    print("  - If WRONG OOD-dist is similar to or LOWER than TRUE OOD-dist, the i->2")
    print("    preference reflects a genuine (if surprising) similarity to layer 2's")
    print("    real distribution -- not a shortcut.")
    print("  - If WRONG OOD-dist is MARKEDLY HIGHER than TRUE OOD-dist despite winning")
    print("    on FVE, that's a shortcut signature: the MLP found a way to score well")
    print("    on the training objective without genuinely resembling layer 2's real")
    print("    activations -- the same category of problem the reliability paper warns")
    print("    about for TLT, but arising here from stitcher EXPRESSIVENESS alone.")


if __name__ == "__main__":
    main()
