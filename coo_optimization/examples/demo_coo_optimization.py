# %%

"""
Example: Using Canine Olfactory Optimizer (COO)
==================================================

This demo runs the COO algorithm on the Rastrigin test function
and plots its convergence curve. It uses the Surrogate Ensemble
and adaptive mechanisms built into the optimizer.

Requirements
------------
- numpy
- matplotlib
- joblib
- scikit-learn

Run this example:
-----------------
    python examples/demo_coo_optimization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from coo_optimization import CanineOlfactoryOptimization


# ============================================================
# 1. Define the benchmark objective function
# ============================================================

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (global minimum at x=0, f=0)."""
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# ============================================================
# 2. Set bounds and initialize the optimizer
# ============================================================

dim = 10
bounds = [(-5.12, 5.12)] * dim

optimizer = CanineOlfactoryOptimization(
    bounds=bounds,
    n_packs=3,
    init_pack_size=15,
    max_iterations=120,
    surrogate_enabled=True,
    surrogate_kind='ensemble',
    surrogate_retrain_freq=5,
    use_elitist=True,
    use_gradient=True,
    use_adaptive_pack=True,
    verbose=True,
    random_state=42
)

# ============================================================
# 3. Run optimization
# ============================================================

best_pos, best_fit, history, diagnostics = optimizer.optimize(rastrigin)

print("\nBest position found:")
print(best_pos)
print(f"Best fitness: {best_fit:.6f}")
print("Diagnostics:", diagnostics)


# ============================================================
# 4. Plot convergence history
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history, label='COO Convergence', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness (Rastrigin)')
plt.title('COO Optimization Convergence')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 5. (Optional) Compare multiple runs / parameter tuning
# ============================================================

def tune_runs(n_runs=3):
    results = []
    for seed in range(n_runs):
        opt = CanineOlfactoryOptimization(
            bounds=bounds,
            n_packs=3,
            init_pack_size=15,
            max_iterations=100,
            surrogate_enabled=True,
            random_state=seed,
            verbose=False
        )
        _, best, hist, _ = opt.optimize(rastrigin)
        results.append((seed, best, hist))
    return results


# Run multiple trials
runs = tune_runs(n_runs=3)

# Plot all runs
plt.figure(figsize=(8, 5))
for seed, best, hist in runs:
    plt.plot(hist, label=f'Run {seed} (best={best:.2f})')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('COO Multiple Runs (Tuning Comparison)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %%
