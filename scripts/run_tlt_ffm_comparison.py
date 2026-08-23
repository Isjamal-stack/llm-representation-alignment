"""
Phase A, Step 3: TLT vs FFM comparison -- the actual core question from
the reliability paper, tested here in audio for the first time.

Stitches --source_model's layer i to --target_model's layer i (defaults to
the SAME model for both, i.e. self-stitching -- the reliability paper's
original, cleanest setting; pass different values for a genuine cross-model
test, e.g. --source_model wav2vec2-base --target_model hubert-base),
comparing three ways of training the stitch layer:

  - LFM       : label-free, matches at the stitch point (our existing
                Phase A baseline).
  - TLT       : Task Loss Training -- trains directly on a downstream
                classification task, from random init.
  - FFM->TLT  : two-stage recipe -- pretrain to match the TARGET's FINAL
                features (label-free), then fine-tune with TLT.

For each, we report (a) downstream classification accuracy and (b) an
OOD proxy distance at the stitch layer (how far the stitched activation
sits from the TARGET's own real activation distribution at that layer).
The reliability paper's core finding, if it replicates in audio, would
show TLT achieving good accuracy but a LARGER OOD distance than LFM or
FFM->TLT -- i.e. TLT "cheats" by pushing the stitched layer's activations
out of distribution rather than genuinely aligning with the target's real
representation.

SCOPE NOTES (read before interpreting results):
  - Requires --source_model and --target_model to have the SAME number of
    layers (position i in source maps to position i in target). True for
    all three registered models (wav2vec2-base, hubert-base,
    wavlm-base-plus -- all 12 layers), but not checked beyond that.
  - The downstream task defaults to a 3-class chapter-id classification on
    the same LibriSpeech dummy split used elsewhere in Phase A -- a
    pipeline-validation-scale task, not a proper SUPERB benchmark. See
    get_labeled_audio_batch's docstring in extraction_utils.py.
  - The OOD check uses a fast, training-free distributional-distance
    proxy (src/ood_detector.py: simple_distribution_distance), not the
    full energy-based detector (also in ood_detector.py, unused here) --
    that needs an auxiliary OOD dataset not yet chosen.

Usage (self-stitch, original setting):
    python scripts/run_tlt_ffm_comparison.py \
        --source_model wav2vec2-base --target_model wav2vec2-base \
        --layers 3 6 9 --num_examples 64

Usage (cross-model, new):
    python scripts/run_tlt_ffm_comparison.py \
        --source_model wav2vec2-base --target_model hubert-base \
        --layers 3 6 9 --num_examples 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from audio_models import load_model
from extraction_utils import get_labeled_audio_batch, extract_hidden_states_chunked
from stitching import StitchLayer, TemporalResample, lfm_loss, ffm_loss, task_loss, load_direct_matching_init
from ood_detector import simple_distribution_distance


def mean_pool(feat):
    return feat.mean(dim=1)  # (B, T, C) -> (B, C)


def train_with_early_stop(params, step_fn, epochs, lr, patience):
    """Generic Adam training loop with early stopping on a loss value
    returned by step_fn(). Shared by the probe and LFM loops below."""
    opt = torch.optim.AdamW(params, lr=lr)
    best_loss, patience_counter = float("inf"), 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = step_fn()
        loss.backward()
        opt.step()
        lv = loss.item()
        if lv < best_loss - 1e-8:
            best_loss, patience_counter = lv, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_loss


def train_with_tracking(params, forward_fn, loss_from_output_fn, ood_target_feat,
                         epochs, lr, patience, eval_every=10):
    """
    Like train_with_early_stop, but also records the OOD proxy distance
    of the stitched activation every `eval_every` epochs (and at epoch 0,
    before any training) -- not just a single before/after snapshot.

    Added specifically to test whether FFM->TLT's OOD distance starts low
    (near LFM's safe level) and drifts upward during the TLT fine-tuning
    stage, or was never actually low to begin with. A snapshot comparison
    alone can't distinguish these two very different explanations.

    forward_fn() -> the stitched intermediate activation (B, T, C), WITH
                     gradients (used both for the loss and, detached, for
                     the periodic OOD check).
    loss_from_output_fn(stitched) -> scalar loss for backprop, computed
                     from that same stitched activation (e.g. by running
                     it through remaining_layers + probe + task_loss, or
                     remaining_layers + ffm_loss).

    Returns: (final_loss, trajectory) where trajectory is a list of
    (epoch, ood_distance) tuples.
    """
    opt = torch.optim.AdamW(params, lr=lr)
    best_loss, patience_counter = float("inf"), 0
    trajectory = []

    with torch.no_grad():
        stitched0 = forward_fn()
        trajectory.append((0, simple_distribution_distance(stitched0, ood_target_feat)))

    final_loss = best_loss
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        stitched = forward_fn()
        loss = loss_from_output_fn(stitched)
        loss.backward()
        opt.step()
        lv = loss.item()
        final_loss = lv

        if epoch % eval_every == 0:
            with torch.no_grad():
                stitched_eval = forward_fn()
                trajectory.append((epoch, simple_distribution_distance(stitched_eval, ood_target_feat)))

        if lv < best_loss - 1e-8:
            best_loss, patience_counter = lv, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return final_loss, trajectory


def train_linear_probe(final_feats, labels_t, num_classes, device, epochs, lr, patience):
    probe = nn.Linear(final_feats.shape[-1], num_classes).to(device)
    pooled = mean_pool(final_feats.to(device))
    train_with_early_stop(
        probe.parameters(),
        lambda: task_loss(probe(pooled), labels_t),
        epochs, lr, patience,
    )
    with torch.no_grad():
        acc = (probe(pooled).argmax(dim=-1) == labels_t).float().mean().item()
    return probe, acc


def eval_stitched(stitch, remaining_layers, probe, src_feat, real_tgt_feat, labels_t):
    with torch.no_grad():
        stitched = stitch(src_feat)
        final = remaining_layers(stitched)
        logits = probe(mean_pool(final))
        acc = (logits.argmax(dim=-1) == labels_t).float().mean().item()
        ood = simple_distribution_distance(stitched, real_tgt_feat)
    return acc, ood


def print_trajectory(name, trajectory):
    points = "  ".join(f"ep{ep}:{dist:.4f}" for ep, dist in trajectory)
    print(f"    {name} OOD trajectory: {points}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_model", default="wav2vec2-base",
                     help="Any key registered in audio_models.py (wav2vec2-base, hubert-base, "
                          "wavlm-base-plus, whisper-base, clap-htsat-unfused, audiomae, patchtst, "
                          "...). No choices= restriction here -- load_model() validates and lists "
                          "valid options if the key is wrong.")
    ap.add_argument("--target_model", default="wav2vec2-base",
                     help="Defaults to the same value as --source_model (self-stitch). "
                          "Pass a different model for a genuine cross-model test.")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--label_column", default="chapter_id",
                     help="NOTE: 'speaker_id' is degenerate (only 1 unique value) in the "
                          "default hf-internal-testing/librispeech_asr_dummy split -- "
                          "'chapter_id' (3 classes) is used by default instead.")
    ap.add_argument("--dataset_name", default="hf-internal-testing/librispeech_asr_dummy")
    ap.add_argument("--dataset_config", default="clean")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--tlt_epochs", type=int, default=200)
    ap.add_argument("--ffm_epochs", type=int, default=200)
    ap.add_argument("--lfm_epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=10,
                     help="Record the OOD trajectory every this many epochs during "
                          "TLT and FFM-stage-2 training.")
    ap.add_argument("--extract_batch_size", type=int, default=8)
    ap.add_argument("--max_audio_samples", type=int, default=160000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    is_self_stitch = (args.source_model == args.target_model)
    print(f"Loading source model: {args.source_model} on {args.device}")
    source_model = load_model(args.source_model, device=args.device)
    if is_self_stitch:
        print("(target_model == source_model -> self-stitch mode)")
        target_model = source_model
    else:
        print(f"Loading target model: {args.target_model} on {args.device}")
        target_model = load_model(args.target_model, device=args.device)

    if source_model.num_layers() != target_model.num_layers():
        print(f"ERROR: source has {source_model.num_layers()} layers, target has "
              f"{target_model.num_layers()} layers. This script assumes matching layer counts.")
        sys.exit(1)

    print(f"Loading {args.num_examples} labeled examples (label_column='{args.label_column}')...")
    waveforms, labels, label2idx = get_labeled_audio_batch(
        args.num_examples, label_column=args.label_column, max_samples=args.max_audio_samples,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config, split=args.split,
    )
    print(f"  {len(label2idx)} unique classes found: {list(label2idx.keys())}")
    num_classes = len(label2idx)
    labels_t = torch.tensor(labels, device=args.device)

    print("Extracting hidden states for the TARGET model...")
    target_input = target_model.preprocess(waveforms)
    target_hidden = extract_hidden_states_chunked(target_model, target_input, chunk_size=args.extract_batch_size)
    final_feats_cpu = target_hidden[target_model.num_layers()]

    if is_self_stitch:
        source_hidden = target_hidden
    else:
        print("Extracting hidden states for the SOURCE model...")
        source_input = source_model.preprocess(waveforms)
        source_hidden = extract_hidden_states_chunked(source_model, source_input, chunk_size=args.extract_batch_size)

        src_T = source_hidden[1].shape[1]
        tgt_T = target_hidden[1].shape[1]
        if src_T != tgt_T:
            print(f"NOTE: source T={src_T} != target T={tgt_T}; resampling source's time axis to match.")
            resampler = TemporalResample(target_frames=tgt_T)
            source_hidden = tuple(resampler(h) for h in source_hidden)
        else:
            print(f"Source and target frame counts match (T={src_T}); no resampling needed.")

    print("\nTraining linear probe on the TARGET model's REAL final-layer features "
          "(this becomes the frozen downstream task head TLT/FFM are evaluated through)...")
    probe, probe_acc = train_linear_probe(
        final_feats_cpu, labels_t, num_classes, args.device,
        epochs=args.probe_epochs, lr=args.lr, patience=args.patience,
    )
    print(f"  Probe training accuracy (on real, unstitched features): {probe_acc:.2%}")
    for p in probe.parameters():
        p.requires_grad_(False)

    stitch_label = "Self-stitching" if is_self_stitch else f"Stitching {args.source_model} \u2192 {args.target_model}"
    results = []
    for layer_i in args.layers:
        print(f"\n=== {stitch_label} at layer {layer_i} ===")
        src_feat_cpu = source_hidden[layer_i]
        tgt_feat_cpu = target_hidden[layer_i]

        src_feat = src_feat_cpu.to(args.device)
        tgt_feat = tgt_feat_cpu.to(args.device)
        real_final = final_feats_cpu.to(args.device)
        remaining_layers = target_model.head_from_layer(layer_i)

        # ---- LFM baseline ----
        print(f"  Training LFM...", end=" ", flush=True)
        stitch_lfm = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind="linear").to(args.device)
        load_direct_matching_init(stitch_lfm, src_feat_cpu, tgt_feat_cpu)
        train_with_early_stop(
            stitch_lfm.parameters(), lambda: lfm_loss(stitch_lfm(src_feat), tgt_feat),
            args.lfm_epochs, args.lr, args.patience,
        )
        acc_lfm, ood_lfm = eval_stitched(stitch_lfm, remaining_layers, probe, src_feat, tgt_feat, labels_t)
        print("done.")

        # ---- TLT: random init, trained directly on the downstream task ----
        print(f"  Training TLT...", end=" ", flush=True)
        stitch_tlt = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind="linear").to(args.device)
        _, tlt_trajectory = train_with_tracking(
            stitch_tlt.parameters(),
            lambda: stitch_tlt(src_feat),
            lambda stitched: task_loss(probe(mean_pool(remaining_layers(stitched))), labels_t),
            tgt_feat, args.tlt_epochs, args.lr, args.patience, eval_every=args.eval_every,
        )
        acc_tlt, ood_tlt = eval_stitched(stitch_tlt, remaining_layers, probe, src_feat, tgt_feat, labels_t)
        print("done.")
        print_trajectory("TLT", tlt_trajectory)

        # ---- FFM (pretrain, label-free) -> TLT (fine-tune) ----
        print(f"  Training FFM (stage 1/2, label-free pretrain)...", end=" ", flush=True)
        stitch_ffm = StitchLayer(src_feat.shape[-1], tgt_feat.shape[-1], kind="linear").to(args.device)
        train_with_early_stop(
            stitch_ffm.parameters(),
            lambda: ffm_loss(remaining_layers(stitch_ffm(src_feat)), real_final),
            args.ffm_epochs, args.lr, args.patience,
        )
        print("done.")
        print(f"  Training FFM (stage 2/2, TLT fine-tune)...", end=" ", flush=True)
        _, ffm_stage2_trajectory = train_with_tracking(
            stitch_ffm.parameters(),
            lambda: stitch_ffm(src_feat),
            lambda stitched: task_loss(probe(mean_pool(remaining_layers(stitched))), labels_t),
            tgt_feat, args.tlt_epochs, args.lr, args.patience, eval_every=args.eval_every,
        )
        acc_ffm, ood_ffm = eval_stitched(stitch_ffm, remaining_layers, probe, src_feat, tgt_feat, labels_t)
        print("done.")
        print_trajectory("FFM stage-2", ffm_stage2_trajectory)

        results.append((layer_i, acc_lfm, ood_lfm, acc_tlt, ood_tlt, acc_ffm, ood_ffm))
        print(f"  LFM:      accuracy={acc_lfm:.2%}  OOD-distance={ood_lfm:.4f}")
        print(f"  TLT:      accuracy={acc_tlt:.2%}  OOD-distance={ood_tlt:.4f}")
        print(f"  FFM->TLT: accuracy={acc_ffm:.2%}  OOD-distance={ood_ffm:.4f}")

    print("\n=== Summary ===")
    print(f"{'Layer':>6} {'LFM acc':>9} {'LFM OOD':>9} {'TLT acc':>9} {'TLT OOD':>9} {'FFM acc':>9} {'FFM OOD':>9}")
    for layer_i, acc_lfm, ood_lfm, acc_tlt, ood_tlt, acc_ffm, ood_ffm in results:
        print(f"{layer_i:>6} {acc_lfm:>9.2%} {ood_lfm:>9.4f} {acc_tlt:>9.2%} {ood_tlt:>9.4f} "
              f"{acc_ffm:>9.2%} {ood_ffm:>9.4f}")

    print("\nExpected pattern (per the reliability paper, if it replicates in audio):")
    print("  TLT should show HIGH accuracy but also a HIGHER OOD distance than LFM/FFM->TLT.")
    print("  FFM->TLT should show accuracy close to TLT's, but OOD distance closer to LFM's.")
    print("  If TLT's OOD distance is NOT meaningfully higher, that itself is a real finding --")
    print("  it would suggest this failure mode doesn't replicate in audio the way it does in vision.")


if __name__ == "__main__":
    main()
