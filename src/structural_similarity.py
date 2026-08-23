"""
Structural similarity baselines -- computed directly on cached activations,
no stitch-layer training required. Used to cross-check whether a pattern
seen in stitching-based (functional) similarity is genuine or a metric
artifact, per Kornblith et al. 2019 / the reliability paper's Section 2.2.
"""
import torch


def linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Linear CKA (Kornblith et al. 2019, as used in the reliability paper).

    A, B: (N, C) activation matrices, already flattened over batch*time,
    with the SAME N (i.e. computed from the same set of examples/frames,
    just possibly different channel counts C). Column-centers both before
    computing:

        LCKA(A, B) = ||B^T A||_F^2 / (||A^T A||_F * ||B^T B||_F)

    Returns a value in [0, 1]: 1.0 means the two representations' pairwise
    example-similarity structures are identical; 0 means unrelated. Unlike
    the stitching-based scores in stitching.py, this requires no training
    and reflects pure geometric/statistical structure, not "can a small
    trainable map turn one into the other."
    """
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    hsic = torch.norm(B.T @ A, p="fro") ** 2
    norm_a = torch.norm(A.T @ A, p="fro")
    norm_b = torch.norm(B.T @ B, p="fro")
    denom = norm_a * norm_b
    if denom < 1e-12:
        return 0.0
    return (hsic / denom).item()


def flatten_for_cka(feat: torch.Tensor) -> torch.Tensor:
    """(B, T, C) -> (B*T, C), the shape linear_cka expects."""
    B, T, C = feat.shape
    return feat.reshape(B * T, C)
