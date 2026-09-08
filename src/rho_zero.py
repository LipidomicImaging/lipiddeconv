"""Reusable implementation of the frozen v47/v48 rho_zero certificate.

The implementation below preserves the historical data-only semantics:
rho_zero is the normalized increase in optimal reconstruction loss when one
candidate is forced to zero.  The certificate uses post-hoc identity weights,
not production ISTA channel weighting.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import nnls


EPSILON_Q = 1e-15


def identity_weights(A: np.ndarray) -> np.ndarray:
    """Return the frozen primary post-hoc certificate weights (all ones)."""
    return np.ones(A.shape[0], dtype=np.float64)


def solve_nonnegative_lasso(
    A: np.ndarray, b: np.ndarray, penalty: float, iterations: int = 10000
) -> np.ndarray:
    """Exact historical nonnegative FISTA helper retained for shared callers."""
    if penalty == 0:
        return nnls(A, b, maxiter=10 * A.shape[1])[0]
    gram = A.T @ A
    atb = A.T @ b
    vector = np.ones(A.shape[1], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(100):
        vector = gram @ vector
        vector /= max(np.linalg.norm(vector), 1e-15)
    lipschitz = max(float(vector @ gram @ vector), 1e-15)
    step = 0.98 / lipschitz
    x = np.zeros(A.shape[1], dtype=np.float64)
    y = x.copy()
    momentum = 1.0
    for _ in range(iterations):
        x_next = np.maximum(y - step * (gram @ y - atb) - step * penalty, 0.0)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        y = x_next + ((momentum - 1.0) / next_momentum) * (x_next - x)
        if np.linalg.norm(x_next - x) <= 1e-10 * max(np.linalg.norm(x), 1.0):
            x = x_next
            break
        x = x_next
        momentum = next_momentum
    return x


def rho_zero_from_weighted_case(
    A: np.ndarray,
    b: np.ndarray,
    candidate_index: int,
    weights: np.ndarray | None = None,
    epsilon_q: float = EPSILON_Q,
) -> dict[str, float]:
    """Compute the historical rho_zero and its shared necessity quantities.

    A is channels-by-candidates and b is a single weighted channel vector.
    The unconstrained optimum and leave-one-component-out optimum are both
    nonnegative NNLS solutions, exactly as in historical profile_path_1d().
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if weights is None:
        weights = identity_weights(A)
    weights = np.asarray(weights, dtype=np.float64)
    Aw = A * weights[:, None]
    bw = b * weights
    j = int(candidate_index)
    if not 0 <= j < Aw.shape[1]:
        raise IndexError(f"candidate index outside library: {j}")
    x_star, residual = nnls(Aw, bw, maxiter=10 * Aw.shape[1])
    keep = np.arange(Aw.shape[1]) != j
    _, deleted_residual = nnls(Aw[:, keep], bw, maxiter=10 * Aw.shape[1])
    q_star = float(residual * residual)
    q_deleted = float(deleted_residual * deleted_residual)
    signal = float(bw @ bw) + float(epsilon_q)
    delta_q = max(0.0, q_deleted - q_star)
    return {
        "rho_zero": max(0.0, delta_q / signal),
        "necessity_signal": delta_q / max(signal, float(epsilon_q)),
        "necessity_fit": delta_q / max(q_star, float(epsilon_q)),
        "q_star": q_star,
        "q_deleted": q_deleted,
        "signal_norm2_plus_epsilon": signal,
        "x_star_candidate": float(x_star[j]),
    }
