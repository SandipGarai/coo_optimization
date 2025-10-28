"""
Unit tests for Canine Olfactory Optimization (COO)
==================================================

These tests ensure correct functionality of:
 - Initialization
 - Optimization convergence
 - Surrogate model integration
 - Caching and diagnostics

Run with:
    pytest -v tests/test_coo.py
"""

import numpy as np
import pytest
from coo_optimization import CanineOlfactoryOptimization, SurrogateEnsemble

# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


@pytest.fixture
def sphere_function():
    """Simple convex objective: Sphere."""
    return lambda x: np.sum(x ** 2)


@pytest.fixture
def rastrigin_function():
    """Non-convex test function."""
    return lambda x: 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


@pytest.fixture
def simple_bounds():
    """Bounds for 5D test space."""
    return [(-5.12, 5.12)] * 5


# ------------------------------------------------------------
# Tests: Surrogate Ensemble
# ------------------------------------------------------------

def test_surrogate_ensemble_fit_predict(simple_bounds):
    X = np.random.uniform(-5, 5, (50, 5))
    y = np.sum(X ** 2, axis=1)

    model = SurrogateEnsemble(kind="ensemble", random_state=42)
    model.fit(X, y)
    preds = model.predict(X[:5])

    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


# ------------------------------------------------------------
# Tests: COO core functionality
# ------------------------------------------------------------

def test_coo_initialization(simple_bounds):
    opt = CanineOlfactoryOptimization(bounds=simple_bounds)
    assert opt.bounds.shape == (5, 2)
    assert opt.dim == 5
    assert np.all(opt.lower < opt.upper)


def test_coo_optimize_converges(sphere_function, simple_bounds):
    opt = CanineOlfactoryOptimization(
        bounds=simple_bounds,
        n_packs=2,
        init_pack_size=8,
        max_iterations=30,
        surrogate_enabled=False,
        verbose=False,
        random_state=0
    )

    best_pos, best_fit, history, diag = opt.optimize(sphere_function)

    # Convergence checks
    assert isinstance(best_pos, np.ndarray)
    assert np.isscalar(best_fit)
    assert len(history) == 30
    assert best_fit < 10.0  # should get near 0 for Sphere
    assert diag["iterations"] == 30


def test_coo_with_surrogate(rastrigin_function, simple_bounds):
    opt = CanineOlfactoryOptimization(
        bounds=simple_bounds,
        n_packs=3,
        init_pack_size=10,
        max_iterations=20,
        surrogate_enabled=True,
        surrogate_retrain_freq=5,
        surrogate_min_samples=10,
        surrogate_kind="ensemble",
        verbose=False,
        random_state=1
    )

    best_pos, best_fit, history, diag = opt.optimize(rastrigin_function)

    # Verify surrogate worked
    assert len(history) == 20
    assert np.isfinite(best_fit)
    assert diag["cache_size"] > 0


def test_caching_mechanism(simple_bounds, sphere_function):
    opt = CanineOlfactoryOptimization(bounds=simple_bounds)
    x = np.random.uniform(-1, 1, 5)

    val1 = opt._cached_eval(x, sphere_function)
    val2 = opt._cached_eval(x, sphere_function)

    # Should hit cache, so second call should not add new entries
    assert len(opt.evaluation_cache) == 1
    assert np.isclose(val1, val2)


def test_gradient_computation(simple_bounds, sphere_function):
    opt = CanineOlfactoryOptimization(bounds=simple_bounds)
    x = np.random.uniform(-1, 1, 5)

    grad = opt._compute_numerical_gradient(x, sphere_function)
    assert grad.shape == (5,)
    assert np.all(np.isfinite(grad))


def test_elitist_exchange(simple_bounds, sphere_function):
    opt = CanineOlfactoryOptimization(bounds=simple_bounds, use_elitist=True)
    packs = opt._initialize_packs()
    pack_bests = [(1.0, np.random.uniform(-1, 1, 5)) for _ in range(3)]

    opt._elitist_exchange(packs, pack_bests, sphere_function)
    for pack in packs:
        assert pack["positions"].shape[1] == 5
        assert np.all(np.isfinite(pack["positions"]))


# ------------------------------------------------------------
# Edge case and regression tests
# ------------------------------------------------------------

def test_invalid_fit_predict_raises():
    model = SurrogateEnsemble()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 5)))


def test_optimizer_raises_without_fit(simple_bounds):
    opt = CanineOlfactoryOptimization(bounds=simple_bounds)
    with pytest.raises(TypeError):
        opt.optimize(None)
