import math

import numpy as np
import pytest

from pdt_oc.training import lambda_warmup, sample_collocation_points


def test_lambda_warmup_clamps_below_range() -> None:
    assert lambda_warmup(-5, total=50, start=0.1, end=1.0) == 0.1
    assert lambda_warmup(0, total=50, start=0.1, end=1.0) == 0.1


def test_lambda_warmup_clamps_above_range() -> None:
    assert lambda_warmup(50, total=50, start=0.1, end=1.0) == 1.0
    assert lambda_warmup(120, total=50, start=0.1, end=1.0) == 1.0


def test_lambda_warmup_is_linear_at_midpoint() -> None:
    mid = lambda_warmup(25, total=50, start=0.1, end=1.0)
    assert math.isclose(mid, 0.55, rel_tol=1e-12)


def test_collocation_points_lie_strictly_inside_interval() -> None:
    pts = sample_collocation_points((0.0, 10.0), n=100)
    assert pts.shape == (100,)
    assert pts[0] > 0.0
    assert pts[-1] < 10.0


def test_collocation_points_are_strictly_increasing_when_uniform() -> None:
    pts = sample_collocation_points((0.0, 10.0), n=100, jitter=False)
    diffs = np.diff(pts)
    assert np.all(diffs > 0)


def test_collocation_jitter_is_reproducible_with_seed() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    pts_a = sample_collocation_points((0.0, 10.0), n=100, jitter=True, rng=rng_a)
    pts_b = sample_collocation_points((0.0, 10.0), n=100, jitter=True, rng=rng_b)
    assert np.array_equal(pts_a, pts_b)


def test_collocation_rejects_invalid_span() -> None:
    with pytest.raises(ValueError):
        sample_collocation_points((5.0, 5.0), n=10)
    with pytest.raises(ValueError):
        sample_collocation_points((5.0, 1.0), n=10)


def test_collocation_returns_empty_array_when_n_is_zero() -> None:
    pts = sample_collocation_points((0.0, 10.0), n=0)
    assert pts.size == 0
