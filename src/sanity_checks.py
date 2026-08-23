"""
The two cheap, mandatory checks from the reliability paper. Run these
BEFORE any large sweep -- if they fail, the pipeline (not the audio
hypothesis) is almost certainly the problem.

1. Self-stitch identity check: within one model, layer i stitched to
   itself should be the best match for layer i (highest functional
   similarity / lowest distance among all candidate source layers).

2. Cross-model correspondence check: across two independently
   initialized/trained copies of the SAME architecture, the
   architecturally corresponding layer should be the best match.
"""
import numpy as np


def similarity_matrix(scores: dict, num_source_layers: int, num_target_layers: int):
    """
    scores: dict[(src_layer, tgt_layer)] -> scalar similarity (higher = more similar)
    Returns an (num_source_layers, num_target_layers) numpy array.
    """
    mat = np.zeros((num_source_layers, num_target_layers))
    for (i, j), v in scores.items():
        mat[i, j] = v
    return mat


def self_stitch_check(sim_matrix: np.ndarray) -> dict:
    """
    sim_matrix[i, j] = similarity of source layer i stitched to target layer j,
    where source and target are the SAME model. Checks whether argmax of each
    row/col lands on the diagonal.
    """
    n = sim_matrix.shape[0]
    correct = 0
    failures = []
    for i in range(n):
        best_j = int(np.argmax(sim_matrix[i]))
        if best_j == i:
            correct += 1
        else:
            failures.append((i, best_j, sim_matrix[i, best_j], sim_matrix[i, i]))
    return {
        "accuracy": correct / n,
        "failures": failures,  # (true_layer, incorrectly_preferred_layer, its_score, true_score)
    }


def cross_model_correspondence_check(sim_matrix: np.ndarray) -> dict:
    """
    Same shape/interpretation as self_stitch_check, but sim_matrix here
    comes from stitching model A (source) to an independently trained
    copy of the same architecture, model B (target). Correct = argmax
    of row i is column i (architecturally corresponding layer).
    """
    return self_stitch_check(sim_matrix)  # identical logic, different data source


def print_report(name: str, result: dict, layer_names=None):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {result['accuracy']:.2%}")
    if result["failures"]:
        print("Failures (true_layer -> preferred_layer, preferred_score vs true_score):")
        for true_i, pref_j, pref_score, true_score in result["failures"]:
            tn = layer_names[true_i] if layer_names else true_i
            pn = layer_names[pref_j] if layer_names else pref_j
            print(f"  layer {tn} -> preferred {pn}  ({pref_score:.4f} vs {true_score:.4f})")
    else:
        print("No failures. Pipeline passes the sanity check.")
