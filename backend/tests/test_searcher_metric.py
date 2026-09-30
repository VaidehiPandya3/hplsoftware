"""Searcher's cosine metric.

The point of cosine mode is to separate by direction, not magnitude. The test
that matters here is a case L2 and cosine must disagree on: two points on the
same ray as the query but at very different distances, plus one point at a
slightly different angle but close in absolute distance. L2 picks by distance,
cosine picks by angle — if both picked the same neighbour, the mode would be a
no-op wearing a new flag.

Also checks the L2-equivalent distance conversion (2 - 2*cos) is what feeds
vote()'s sqrt(distance), since that's what lets vote()/margin/neighbor_distance
need no changes for this mode.
"""

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from assign_hpc_clusters import Searcher  # noqa: E402


def test_cosine_prefers_direction_over_magnitude():
    # Query points along +x. `near_but_off_angle` is closer in raw distance;
    # `far_but_same_direction` is farther away but exactly on the query's ray.
    reference = np.array([
        [1.0, 0.05],   # near_but_off_angle: small angle, close in L2
        [50.0, 0.0],   # far_but_same_direction: same direction, far in L2
    ], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    l2_idx, _ = Searcher(reference, metric="l2").search(query, k=1)
    cos_idx, _ = Searcher(reference, metric="cosine").search(query, k=1)

    assert l2_idx[0, 0] == 0, "L2 should pick the closer point regardless of angle"
    assert cos_idx[0, 0] == 1, "cosine should pick the same-direction point regardless of distance"


def test_cosine_distance_matches_the_l2_equivalent_formula():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((200, 16)).astype(np.float32)
    query = rng.standard_normal((10, 16)).astype(np.float32)

    searcher = Searcher(reference, metric="cosine")
    idx, dist = searcher.search(query, k=5)

    ref_n = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    q_n = query / np.linalg.norm(query, axis=1, keepdims=True)
    expected_cos = np.einsum("qd,qkd->qk", q_n, ref_n[idx])
    expected_dist = 2.0 - 2.0 * expected_cos

    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)
    # Squared-Euclidean-equivalent distances between unit vectors are in [0, 4].
    assert (dist >= -1e-6).all() and (dist <= 4 + 1e-6).all()


def test_cosine_ignores_scale():
    """Scaling every reference vector by an arbitrary positive factor must not
    change which one is nearest under cosine — the whole premise of the mode."""
    rng = np.random.default_rng(1)
    reference = rng.standard_normal((50, 8)).astype(np.float32)
    query = rng.standard_normal((5, 8)).astype(np.float32)

    idx_a, _ = Searcher(reference, metric="cosine").search(query, k=3)
    scaled = reference * rng.uniform(0.1, 100.0, size=(50, 1)).astype(np.float32)
    idx_b, _ = Searcher(scaled, metric="cosine").search(query, k=3)

    np.testing.assert_array_equal(idx_a, idx_b)


def test_l2_metric_is_unchanged():
    """Regression guard: adding the metric param must not touch existing l2
    behaviour, which is what every previously-measured accuracy number used."""
    rng = np.random.default_rng(2)
    reference = rng.standard_normal((300, 20)).astype(np.float32)
    query = rng.standard_normal((15, 20)).astype(np.float32)

    idx, dist = Searcher(reference, metric="l2").search(query, k=6)

    diffs = reference[idx] - query[:, None, :]
    expected = np.sum(diffs ** 2, axis=2)
    np.testing.assert_allclose(dist, expected, atol=1e-3)
    # device="cpu" explicitly: the default is "auto" now, and on a host with a
    # working GPU faiss that yields "faiss-flat-gpu" — the same exact scan, but
    # this test is about the l2 metric, not about which hardware ran it.
    assert Searcher(reference, device="cpu").backend == "faiss-flat"


def test_unknown_metric_rejected():
    try:
        Searcher(np.zeros((5, 4), np.float32), metric="manhattan")
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "manhattan" in str(e)


def test_k_exceeds_reference_size_padded_with_minus_one_both_metrics():
    reference = np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)
    query = np.ones((1, 3), dtype=np.float32)
    for metric in ("l2", "cosine"):
        idx, dist = Searcher(reference, metric=metric).search(query, k=10)
        assert (idx[0, 4:] == -1).all(), f"metric={metric}"
        assert idx.shape == (1, 10) and dist.shape == (1, 10)


# --- standalone runner ---------------------------------------------------

def main():
    import tempfile
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
