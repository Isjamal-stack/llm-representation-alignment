"""
Unit tests for sanity_checks.py that need only numpy — runnable right now,
with no GPU/network/torch required. These catch logic bugs in the checking
code itself before you ever point it at real model activations on Unity.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sanity_checks import similarity_matrix, self_stitch_check, print_report


def test_perfect_diagonal_passes():
    """If similarity is highest on the diagonal, accuracy should be 100%."""
    n = 6
    sim = np.eye(n) * 1.0 + np.random.RandomState(0).rand(n, n) * 0.3
    np.fill_diagonal(sim, 1.0)  # force diagonal to dominate
    result = self_stitch_check(sim)
    assert result["accuracy"] == 1.0, f"expected 100%, got {result['accuracy']:.2%}"
    assert result["failures"] == []
    print("PASS: perfect diagonal -> 100% accuracy, no failures")


def test_off_diagonal_failure_detected():
    """If a distant layer scores higher than the true diagonal entry,
    the check must catch it (this is exactly the TLM failure mode from
    the reliability paper — the test must be sensitive to it)."""
    sim = np.array([
        [0.5, 0.1, 0.9],   # row 0: layer 2 (0.9) beats true match layer 0 (0.5) -> FAIL
        [0.1, 0.8, 0.1],   # row 1: true match wins -> PASS
        [0.1, 0.1, 0.7],   # row 2: true match wins -> PASS
    ])
    result = self_stitch_check(sim)
    assert result["accuracy"] == pytest_approx(2 / 3), result["accuracy"]
    assert len(result["failures"]) == 1
    true_i, pref_j, pref_score, true_score = result["failures"][0]
    assert true_i == 0 and pref_j == 2
    assert pref_score > true_score
    print("PASS: off-diagonal failure correctly detected and reported")


def pytest_approx(x, tol=1e-9):
    # tiny local helper so this file has zero external test-framework deps
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_similarity_matrix_construction():
    """Sparse dict -> dense matrix conversion should place values correctly
    and default missing entries to 0."""
    scores = {(0, 0): 0.9, (0, 1): 0.2, (1, 1): 0.95}
    mat = similarity_matrix(scores, num_source_layers=2, num_target_layers=2)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == 0.9
    assert mat[0, 1] == 0.2
    assert mat[1, 0] == 0.0  # never set, should default to 0
    assert mat[1, 1] == 0.95
    print("PASS: similarity_matrix places sparse scores correctly")


def test_direct_matching_math_numpy_reference():
    """
    Pure-numpy reference check of the Direct Matching closed-form fit
    (the same math as stitching.direct_matching_init, which needs torch).
    Confirms the least-squares affine fit recovers a known linear relationship
    exactly (up to numerical precision) when one exists in the data —
    this is the sanity check for the math itself, independent of torch/GPU.
    """
    rng = np.random.RandomState(42)
    K, T, C_in, C_out = 20, 5, 8, 6

    # Ground truth affine map: target = source @ W_true.T + b_true
    W_true = rng.randn(C_out, C_in) * 0.5
    b_true = rng.randn(C_out) * 0.1

    source = rng.randn(K, T, C_in)
    target = source @ W_true.T + b_true

    A = source.reshape(-1, C_in)
    A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    B = target.reshape(-1, C_out)
    theta, *_ = np.linalg.lstsq(A_aug, B, rcond=None)
    W_fit, b_fit = theta[:-1].T, theta[-1]

    assert np.allclose(W_fit, W_true, atol=1e-8), "recovered W doesn't match ground truth"
    assert np.allclose(b_fit, b_true, atol=1e-8), "recovered b doesn't match ground truth"

    # Also confirm near-perfect reconstruction on held-out (same-distribution) data
    held_out = rng.randn(K, T, C_in)
    true_out = held_out @ W_true.T + b_true
    pred_out = held_out @ W_fit.T + b_fit
    mse = np.mean((true_out - pred_out) ** 2)
    assert mse < 1e-10, f"reconstruction MSE too high: {mse}"
    print(f"PASS: Direct Matching least-squares math verified (reconstruction MSE={mse:.2e})")


if __name__ == "__main__":
    test_perfect_diagonal_passes()
    test_off_diagonal_failure_detected()
    test_similarity_matrix_construction()
    test_direct_matching_math_numpy_reference()
    print("\nAll offline (no-GPU/no-network) tests passed.")
