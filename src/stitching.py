"""
Core stitching primitives: temporal resampling, stitch layer, and the
stitched model wrapper. Mirrors the vision-paper formalism
(f_<=i -> stitch -> g_>j), adapted for variable-frame-rate audio encoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResample(nn.Module):
    """Resamples the time axis of source features onto the target's frame rate.
    x: (B, T_src, C) -> (B, T_tgt, C)
    """
    def __init__(self, target_frames: int):
        super().__init__()
        self.target_frames = target_frames

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T_src)
        x = F.interpolate(x, size=self.target_frames, mode="linear", align_corners=False)
        return x.transpose(1, 2)  # (B, T_tgt, C)


class StitchLayer(nn.Module):
    """The only trainable component. kind in {'linear', 'mlp', 'conv'}."""
    def __init__(self, in_dim, out_dim, kind="mlp", hidden_dim=None, kernel_size=3):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif kind == "mlp":
            hidden_dim = hidden_dim or max(in_dim, out_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif kind == "conv":
            self.net = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2)
        else:
            raise ValueError(f"unknown stitch layer kind: {kind}")

    def forward(self, x):
        if self.kind == "conv":
            x = x.transpose(1, 2)
            x = self.net(x)
            return x.transpose(1, 2)
        return self.net(x)


class StitchedModel(nn.Module):
    """
    source_extractor(x) -> resample -> stitch -> target_head(.) -> output
    source_extractor and target_head are frozen callables/modules that return
    hidden states at the chosen layer / from the chosen layer onward.
    """
    def __init__(self, source_extractor, target_head, stitch_layer, resampler=None):
        super().__init__()
        self.source_extractor = source_extractor
        self.target_head = target_head
        for p in self.source_extractor.parameters():
            p.requires_grad_(False)
        for p in self.target_head.parameters():
            p.requires_grad_(False)
        self.resampler = resampler
        self.stitch_layer = stitch_layer

    def stitched_features(self, x):
        with torch.no_grad():
            src_feat = self.source_extractor(x)
        if self.resampler is not None:
            src_feat = self.resampler(src_feat)
        return self.stitch_layer(src_feat)

    def forward(self, x):
        stitched = self.stitched_features(x)
        return self.target_head(stitched)


# ---------------- Training objectives ----------------

def lfm_loss(stitched_feat, target_feat):
    """Layer Feature Matching: match at the stitch point. Label-free."""
    return F.mse_loss(stitched_feat, target_feat)


def ffm_loss(stitched_final_feat, target_final_feat):
    """Final Feature Matching: match the TARGET's final-layer output. Label-free."""
    return F.mse_loss(stitched_final_feat, target_final_feat)


def task_loss(logits, labels):
    """Task Loss Training: standard supervised loss on the downstream task."""
    return F.cross_entropy(logits, labels)


def fraction_variance_explained(mse_loss: float, target_feat: torch.Tensor) -> float:
    """
    Scale-invariant similarity score: 1 - MSE / Var(target).

    Raw negative MSE is NOT comparable across target layers/models with
    different activation variances -- a low-variance target trivially
    yields low absolute MSE regardless of whether the mapping is
    genuinely meaningful, which can look like spurious high similarity
    for that target. (Concretely: we saw every source layer in a
    cross-model check falsely prefer one particular target layer, purely
    because that layer happened to have ~2-4x lower variance than the
    others -- not because it was actually more similar.)

    This normalizes for that, giving something analogous to R² in
    regression: a value near 1.0 means the stitcher's prediction explains
    nearly all of the target's variance (genuinely high similarity); near
    0 means it explains none (no better than predicting the mean);
    negative values mean the fit is worse than just predicting the
    target's mean.
    """
    target_var = target_feat.var().item()
    if target_var < 1e-12:
        return float("-inf")  # degenerate (near-constant) target; avoid divide-by-zero
    return 1.0 - (mse_loss / target_var)


# ---------------- Direct Matching closed-form init ----------------

def direct_matching_init(source_feats: torch.Tensor, target_feats: torch.Tensor):
    """
    Closed-form affine least-squares fit (Moore-Penrose pseudoinverse),
    used to initialize a linear StitchLayer before LFM/FFM/TLT training.

    source_feats, target_feats: (K, T, C) sampled over K examples,
    already temporally aligned (same T).
    Returns (W, b) to load into an nn.Linear(in_dim, out_dim).
    """
    K, T, C_in = source_feats.shape
    C_out = target_feats.shape[-1]
    A = source_feats.reshape(-1, C_in)
    A = torch.cat([A, torch.ones(A.shape[0], 1, dtype=A.dtype)], dim=1)  # bias column
    B = target_feats.reshape(-1, C_out)
    theta = torch.linalg.lstsq(A, B).solution
    W, b = theta[:-1].T, theta[-1]
    return W, b


def load_direct_matching_init(stitch_layer: StitchLayer, source_feats, target_feats):
    """Convenience: fit + load into a 'linear'-kind StitchLayer's weights."""
    assert stitch_layer.kind == "linear", "closed-form init only applies to linear stitch layers"
    W, b = direct_matching_init(source_feats, target_feats)
    with torch.no_grad():
        stitch_layer.net.weight.copy_(W)
        stitch_layer.net.bias.copy_(b)
    return stitch_layer
