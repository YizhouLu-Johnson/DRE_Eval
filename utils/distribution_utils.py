from typing import Dict, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.optimize import brentq
from scipy.special import gammaln, psi

GAUSSIAN = "gaussian"
DIRICHLET = "dirichlet"


def infer_distribution_family(data_args: Dict) -> str:
    """Infer whether a config stores Gaussian or Dirichlet parameters."""
    if not isinstance(data_args, dict):
        raise TypeError("data_args must be a dict-like object.")
    if "distribution_family" in data_args:
        return str(data_args["distribution_family"]).lower()
    if "numerator_mean" in data_args:
        return GAUSSIAN
    if "numerator_concentration" in data_args:
        return DIRICHLET
    raise ValueError("Could not infer distribution family from data_args keys.")


def gaussian_kl(mu_a, cov_a, mu_b, cov_b) -> float:
    """KL( N(mu_a, cov_a) || N(mu_b, cov_b) )."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    k = mu_a.shape[0]
    cov_b_inv = np.linalg.inv(cov_b)
    term_trace = np.trace(cov_b_inv @ cov_a)
    diff = (mu_b - mu_a).reshape(-1, 1)
    term_quad = float(diff.T @ cov_b_inv @ diff)
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    if sign_a <= 0 or sign_b <= 0:
        raise ValueError("Covariance matrices must be positive definite.")
    return 0.5 * (term_trace + term_quad - k + (logdet_b - logdet_a))


def dirichlet_kl(alpha, beta) -> float:
    """KL( Dir(alpha) || Dir(beta) )."""
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must have the same shape")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Dirichlet concentration parameters must be > 0.")
    sum_alpha = np.sum(alpha)
    sum_beta = np.sum(beta)
    term1 = gammaln(sum_alpha) - np.sum(gammaln(alpha))
    term2 = -gammaln(sum_beta) + np.sum(gammaln(beta))
    term3 = np.sum((alpha - beta) * (psi(alpha) - psi(sum_alpha)))
    return float(term1 + term2 + term3)


def kl_divergence(params_a: Dict, params_b: Dict, family: str) -> float:
    """Compute KL between distribution ``a`` and ``b`` of the same family."""
    family = family.lower()
    if family == GAUSSIAN:
        return gaussian_kl(params_a["mean"], params_a["cov"], params_b["mean"], params_b["cov"])
    if family == DIRICHLET:
        return dirichlet_kl(params_a["concentration"], params_b["concentration"])
    raise ValueError(f"Unsupported distribution family: {family}")


def sample_distribution(family: str, params: Dict, size: int, rng: RandomState) -> np.ndarray:
    """Sample i.i.d. data from a distribution described by ``params``."""
    family = family.lower()
    if family == GAUSSIAN:
        return rng.multivariate_normal(params["mean"], params["cov"], size=size)
    if family == DIRICHLET:
        return rng.dirichlet(params["concentration"], size=size)
    raise ValueError(f"Unsupported distribution family: {family}")


def get_distribution_params(data_args: Dict, role: str) -> Tuple[str, Dict]:
    """
    Extract parameter dict for ``role`` (\"numerator\" or \"denominator\")
    and return (family, params).
    """
    family = infer_distribution_family(data_args)
    key = f"{role}_mean"
    if family == GAUSSIAN:
        params = {
            "mean": np.asarray(data_args[key], dtype=np.float64),
            "cov": np.asarray(data_args[f"{role}_cov"], dtype=np.float64),
        }
        return family, params

    if family == DIRICHLET:
        conc_key = f"{role}_concentration"
        if conc_key not in data_args:
            raise KeyError(f"{conc_key} missing from data_args for Dirichlet distribution.")
        params = {
            "concentration": np.asarray(data_args[conc_key], dtype=np.float64),
        }
        return family, params

    raise ValueError(f"Unsupported distribution family: {family}")


def dirichlet_scale_for_kl(base_alpha: np.ndarray,
                           target_kl: float,
                           initial_scale: float = 0.5,
                           min_scale: float = 1e-6,
                           max_scale: float = 0.999999,
                           tol: float = 1e-8,
                           max_iter: int = 100) -> float:
    """
    Find scale ``c`` in (0, 1) such that KL(Dir(c * base_alpha) || Dir(base_alpha)) = target_kl.
    """
    base_alpha = np.asarray(base_alpha, dtype=np.float64)
    if target_kl <= 0:
        return 1.0

    def fn(scale):
        return dirichlet_kl(scale * base_alpha, base_alpha) - target_kl

    lower = min(initial_scale, max_scale)
    value = fn(lower)
    attempts = 0
    while value < 0 and attempts < max_iter:
        lower *= 0.5
        if lower <= min_scale:
            lower = min_scale
            break
        value = fn(lower)
        attempts += 1
    if value < 0:
        raise ValueError(f"Unable to bracket target KL={target_kl}; too small even at scale={lower}")

    upper = max_scale
    return float(brentq(fn, lower, upper, xtol=tol, maxiter=max_iter))


def dirichlet_translate_for_kl(alpha_start: np.ndarray,
                               direction: np.ndarray,
                               target_kl: float,
                               tol: float = 1e-8,
                               max_iter: int = 100) -> np.ndarray:
    """
    Move along ``direction`` (element-wise positive) from ``alpha_start`` until
    KL(candidate || alpha_start) == target_kl. Returns the candidate concentration vector.
    """
    alpha_start = np.asarray(alpha_start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if np.any(direction == 0):
        direction = direction + 1e-8

    def candidate(scale):
        return alpha_start + scale * direction

    def fn(scale):
        return dirichlet_kl(candidate(scale), alpha_start)

    def objective(scale):
        return fn(scale) - target_kl

    high = 1.0
    val_high = objective(high)
    iters = 0
    while val_high < 0 and iters < max_iter:
        high *= 2.0
        val_high = objective(high)
        iters += 1
        if high > 1e8:
            raise ValueError("Failed to bracket Dirichlet KL target when translating.")

    scale = brentq(objective, 0.0, high, xtol=tol, maxiter=max_iter)
    return candidate(scale)


def dirichlet_translate_reference(alpha_reference: np.ndarray,
                                  direction: np.ndarray,
                                  target_kl: float,
                                  tol: float = 1e-8,
                                  max_iter: int = 100) -> np.ndarray:
    """
    Move away from ``alpha_reference`` along ``direction`` until
    KL(alpha_reference || candidate) == target_kl.
    """
    alpha_reference = np.asarray(alpha_reference, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if np.any(direction == 0):
        direction = direction + 1e-8

    def candidate(scale):
        return alpha_reference + scale * direction

    def objective(scale):
        return dirichlet_kl(alpha_reference, candidate(scale)) - target_kl

    high = 1.0
    val_high = objective(high)
    iters = 0
    while val_high < 0 and iters < max_iter:
        high *= 2.0
        val_high = objective(high)
        iters += 1
        if high > 1e8:
            raise ValueError("Failed to bracket Dirichlet KL target (forward translation).")

    scale = brentq(objective, 0.0, high, xtol=tol, maxiter=max_iter)
    return candidate(scale)
