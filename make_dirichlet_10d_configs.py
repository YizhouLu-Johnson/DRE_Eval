#!/usr/bin/env python3
"""
Generate base Dirichlet-10D configs with:
- fixed P0 = Dir(alpha0)
- variable P1 = Dir(alpha1) such that KL(P0||P1) hits target_kls (10..80)

We parameterize P1 by moving the mean on the simplex + adjusting total concentration:
    p1(t)      = softmax(log p0 + t v)      (mean move; sum=1)
    alpha1(u,t)= c0 * exp(u) * p1(t)        (total concentration scale)

For each target K:
  (A) For a given u, solve t>=0 such that KL(P0||P1)=K  (if feasible)
  (B) Optionally solve u so KL(P1||P0) ~= ratio_target*K
      using a robust scan + bracketing; fallback to u=0 if not solvable.

We store:
  - base_kl_p0_p1 = KL(P0||P1)  (sweep knob you asked for)
  - base_kl_p1_p0 = KL(P1||P0)  (should be similar if match succeeds)

Training convention:
  numerator_concentration   = alpha1 (P1)
  denominator_concentration = alpha0 (P0)
  analytic_kl/target_kl/true_mutual_info = KL(P1||P0)
"""

import argparse
from copy import deepcopy
from time import gmtime, strftime
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gammaln, psi

import make_configs
from make_configs import make_base_config, save_config


# ---------------------------
# Dirichlet KL
# ---------------------------

def dirichlet_kl(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    """KL( Dir(alpha_p) || Dir(alpha_q) )."""
    alpha_p = np.asarray(alpha_p, dtype=np.float64)
    alpha_q = np.asarray(alpha_q, dtype=np.float64)
    if np.any(alpha_p <= 0) or np.any(alpha_q <= 0):
        raise ValueError("Dirichlet parameters must be > 0.")
    term1 = np.sum(gammaln(alpha_q)) - gammaln(np.sum(alpha_q))
    term2 = -np.sum(gammaln(alpha_p)) + gammaln(np.sum(alpha_p))
    term3 = np.sum((alpha_p - alpha_q) * (psi(alpha_p) - psi(np.sum(alpha_p))))
    return float(term1 + term2 + term3)


# ---------------------------
# Helpers
# ---------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def _ones_vector(n_dims: int, value: float) -> List[float]:
    return [float(value)] * n_dims

def balanced_direction(d: int, focus_dim: int = 0) -> np.ndarray:
    """Deterministic balanced direction with sum=0."""
    if not (0 <= focus_dim < d):
        raise ValueError("focus_dim out of range")
    v = np.full(d, -1.0 / (d - 1), dtype=np.float64)
    v[focus_dim] = 1.0
    v /= np.linalg.norm(v)
    return v

def make_alpha1(alpha0: np.ndarray, v: np.ndarray, u: float, t: float) -> np.ndarray:
    """
    alpha1(u,t) = c0*exp(u) * softmax(log p0 + t v)
    """
    c0 = float(np.sum(alpha0))
    p0 = alpha0 / c0
    p1 = softmax(np.log(p0) + t * v)
    c1 = c0 * np.exp(u)
    return c1 * p1


# ---------------------------
# Solve t for KL(P0||P1)=K, robustly
# ---------------------------

def solve_t_for_forward_kl(alpha0: np.ndarray,
                           v: np.ndarray,
                           u: float,
                           target_K: float,
                           t_max: float = 300.0) -> Optional[float]:
    """
    Solve t>=0 such that KL(P0||P1(u,t)) = target_K for fixed u.

    Returns:
      t (float) if solvable, else None.
    """
    if target_K < 0:
        raise ValueError("target_K must be nonnegative")

    def f(t: float) -> float:
        a1 = make_alpha1(alpha0, v, u, t)
        return dirichlet_kl(alpha0, a1) - target_K

    lo = 0.0
    flo = f(lo)

    if abs(flo) < 1e-12:
        return 0.0

    # Try to find hi such that f(lo) and f(hi) have opposite signs.
    hi = 0.2
    fhi = f(hi)

    if flo < 0:
        # Need to increase t until we cross upward (fhi >= 0)
        while fhi < 0:
            hi *= 2.0
            if hi > t_max:
                return None
            fhi = f(hi)
        # Now flo<0 and fhi>=0 -> bracket exists
        sol = root_scalar(f, bracket=[lo, hi], method="brentq", xtol=1e-12)
        return float(sol.root) if sol.converged else None

    else:
        # flo > 0 at t=0: for many u this means target_K is already exceeded.
        # A root may or may not exist for t>0, so search for a sign change.
        # We look for some hi with fhi < 0.
        while fhi > 0:
            hi *= 2.0
            if hi > t_max:
                return None
            fhi = f(hi)

        # Now flo>0 and fhi<=0 -> bracket exists
        sol = root_scalar(f, bracket=[lo, hi], method="brentq", xtol=1e-12)
        return float(sol.root) if sol.converged else None


# ---------------------------
# Solve u to match reverse KL, robustly
# ---------------------------

def solve_u_for_reverse_match(alpha0: np.ndarray,
                             v: np.ndarray,
                             target_K: float,
                             ratio_target: float = 1.0,
                             u_scan_max: float = 6.0,
                             u_scan_points: int = 61) -> Tuple[float, bool]:
    """
    Try to solve u such that after choosing t(u) satisfying KL(P0||P1)=K,
    we also get KL(P1||P0) ~= ratio_target*K.

    Returns:
      (u, matched_flag)
    If matching fails, returns (0.0, False) meaning "use u=0".
    """
    desired = ratio_target * target_K

    def g(u: float) -> Optional[float]:
        t = solve_t_for_forward_kl(alpha0, v, u, target_K)
        if t is None:
            return None
        a1 = make_alpha1(alpha0, v, u, t)
        return dirichlet_kl(a1, alpha0) - desired

    # Scan u around 0 to find a feasible sign change
    us = np.linspace(-u_scan_max, u_scan_max, u_scan_points)
    vals = []
    for u in us:
        gv = g(float(u))
        vals.append(gv)

    # Collect feasible indices
    feasible = [i for i, gv in enumerate(vals) if gv is not None and np.isfinite(gv)]
    if len(feasible) < 2:
        return 0.0, False

    # Look for adjacent feasible points with opposite signs
    for i0, i1 in zip(feasible[:-1], feasible[1:]):
        g0, g1 = vals[i0], vals[i1]
        if g0 == 0:
            return float(us[i0]), True
        if g0 * g1 < 0:
            lo, hi = float(us[i0]), float(us[i1])

            def gg(u: float) -> float:
                out = g(u)
                if out is None:
                    # Should not happen inside bracket if scan was feasible,
                    # but keep it safe.
                    raise RuntimeError("Lost feasibility inside u bracket.")
                return float(out)

            sol = root_scalar(gg, bracket=[lo, hi], method="brentq", xtol=1e-10)
            if sol.converged:
                return float(sol.root), True
            return 0.0, False

    # No sign change found -> can't hit ratio_target exactly; fallback
    return 0.0, False


# ---------------------------
# Schedules
# ---------------------------

def pick_value_from_schedule(x, breakpoints, values):
    assert len(values) == len(breakpoints) + 1, \
        "Need one more value than breakpoints (piecewise constant schedule)."
    for bp, val in zip(breakpoints, values):
        if x < bp:
            return val
    return values[-1]


# ---------------------------
# Config builder
# ---------------------------

def build_single_config(base_config: dict,
                        target_kl_p0_p1: float,
                        sample_size: int,
                        trial_idx: int,
                        n_waymarks: int,
                        n_epochs: int,
                        seed: int,
                        base_concentration: float,
                        focus_dim: int,
                        ratio_target: float,
                        try_match_reverse: bool) -> dict:
    cfg = deepcopy(base_config)
    d = int(cfg["data"]["n_dims"])

    # Fixed P0
    alpha0 = np.array(_ones_vector(d, base_concentration), dtype=np.float64)

    v = balanced_direction(d, focus_dim=focus_dim)

    # Pick u (either matched or 0)
    if try_match_reverse:
        u, matched = solve_u_for_reverse_match(alpha0, v, target_kl_p0_p1, ratio_target=ratio_target)
    else:
        u, matched = 0.0, False

    # Solve t for forward KL (must exist near u=0)
    t = solve_t_for_forward_kl(alpha0, v, u, target_kl_p0_p1)
    if t is None:
        # This should basically never happen at u=0 for K>0,
        # but keep the script from silently producing nonsense.
        raise RuntimeError(
            f"Could not solve t for KL(P0||P1)={target_kl_p0_p1} with u={u}. "
            "Try reducing ratio_target toward 1.0 or increasing base_concentration."
        )

    alpha1 = make_alpha1(alpha0, v, u, t)

    kl_p0_p1 = dirichlet_kl(alpha0, alpha1)
    kl_p1_p0 = dirichlet_kl(alpha1, alpha0)

    cfg["data"]["dataset_name"] = "dirichlet_10d"
    cfg["data"]["data_seed"] = int(seed)
    cfg["data"]["noise_dist_name"] = "dirichlet"
    cfg["data"]["data_dist_name"] = "dirichlet"

    cfg["data"]["linear_combo_alphas"] = np.linspace(0.0, 1.0, n_waymarks + 1).tolist()
    cfg["data"]["initial_waymark_indices"] = list(range(n_waymarks + 1))

    # Denominator is P0 (fixed), numerator is P1
    cfg["data"]["noise_dist_dirichlet_concentration"] = alpha0.tolist()

    cfg["data"]["data_args"] = {
        "distribution_family": "dirichlet",
        "n_samples": int(sample_size),
        "n_dims": d,

        "numerator_concentration": alpha1.tolist(),   # P1
        "denominator_concentration": alpha0.tolist(), # P0

        "base_concentration": float(base_concentration),
        "focus_dim": int(focus_dim),
        "ratio_target": float(ratio_target),
        "try_match_reverse": bool(try_match_reverse),
        "reverse_match_success": bool(matched),

        "move_u": float(u),
        "move_t": float(t),
        "move_direction": v.tolist(),

        # store both directions explicitly
        "base_kl_p0_p1": float(kl_p0_p1),
        "base_kl_p1_p0": float(kl_p1_p0),

        # DRE “training KL” (numerator||denominator)
        "true_mutual_info": float(kl_p1_p0),
        "analytic_kl": float(kl_p1_p0),
        "target_kl": float(kl_p1_p0),

        "train_sample_size": int(sample_size),
        "trial_id": int(trial_idx),
    }

    # Optim / arch
    cfg["optimisation"]["n_batch"] = min(sample_size, 128)
    cfg["optimisation"]["loss_function"] = "logistic"
    cfg["optimisation"]["energy_lr"] = 5e-4
    cfg["optimisation"]["n_epochs"] = int(n_epochs)
    cfg["optimisation"]["patience"] = 100
    cfg["optimisation"]["save_every_x_epochs"] = None

    cfg["architecture"]["network_type"] = "mlp"
    cfg["architecture"]["mlp_hidden_size"] = 256
    cfg["architecture"]["mlp_n_blocks"] = 3

    return cfg


def generate_configs(target_kls: Iterable[float],
                     sample_sizes: Iterable[int],
                     n_trials: int,
                     base_seed: int,
                     time_id: str,
                     base_concentration: float,
                     focus_dim: int,
                     ratio_target: float,
                     try_match_reverse: bool,
                     waymark_breakpoints=None,
                     waymark_counts=None,
                     epoch_breakpoints=None,
                     epoch_values=None,
                     save_root=None,
                     config_dir_name=None):

    make_configs.time_id = time_id
    base_cfg = make_base_config()
    base_cfg["data"]["dataset_name"] = "dirichlet_10d"
    base_cfg["data"]["n_dims"] = 10

    if config_dir_name:
        base_cfg["data"]["config_dir_name"] = config_dir_name
    if save_root:
        base_cfg["data"]["save_dir_root"] = save_root

    configs = []
    idx = 0
    for k in target_kls:
        for n in sample_sizes:
            for trial in range(n_trials):
                seed = base_seed + trial + int(float(k) * 137)
                n_waymarks = pick_value_from_schedule(n, waymark_breakpoints, waymark_counts)
                n_epochs = pick_value_from_schedule(n, epoch_breakpoints, epoch_values)

                cfg = build_single_config(
                    base_cfg,
                    target_kl_p0_p1=float(k),
                    sample_size=int(n),
                    trial_idx=int(trial),
                    n_waymarks=int(n_waymarks),
                    n_epochs=int(n_epochs),
                    seed=int(seed),
                    base_concentration=float(base_concentration),
                    focus_dim=int(focus_dim),
                    ratio_target=float(ratio_target),
                    try_match_reverse=bool(try_match_reverse),
                )
                save_config(cfg, "model", idx)
                configs.append(cfg)
                idx += 1

    print(f"Generated {len(configs)} base Dirichlet configs under time_id={time_id}.")
    return configs



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dirichlet_10d configs with fixed KL gaps.")
    parser.add_argument("--target_kls", type=float, nargs="+",
                        default=[80],
                        help="Target KL(P0||P1) values (nats).")
    parser.add_argument("--sample_sizes", type=int, nargs="+",
                        default=[3000],
                        help="Training sample sizes per trial.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of configs per (KL, sample_size).")
    parser.add_argument("--waymark_breakpoints", type=int, nargs="+", default=[1000],
                        help="Sample-size thresholds for changing waymarks.")
    parser.add_argument("--waymark_counts", type=int, nargs="+", default=[12, 12],
                        help="Waymark counts per region (len=breakpoints+1).")
    parser.add_argument("--epoch_breakpoints", type=int, nargs="+", default=[1000],
                        help="Sample-size thresholds where n_epochs changes.")
    parser.add_argument("--epoch_values", type=int, nargs="+", default=[400, 550],
                        help="Epoch counts in each region.")
    parser.add_argument("--base_seed", type=int, default=474747,
                        help="Seed offset applied to data sampling.")
    parser.add_argument("--time_id", type=str, default=None,
                        help="Identifier appended to save/config dirs.")
    parser.add_argument("--base_concentration", type=float, default=30.0,
                        help="Symmetric concentration of P1.")
    parser.add_argument("--ratio_target", type=float, default=1.0,
                        help="Try to enforce KL(P1||P0) ~= ratio_target * KL(P0||P1).")
    parser.add_argument("--focus_dim", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="dirichlet_10d_kl80",
                        help="Subdirectory under saved_models/ for TDRE runs.")
    parser.add_argument("--config_dir_name", type=str, default="dirichlet_10d_80",
                        help="Subdirectory under configs/ for generated files.")
    parser.add_argument("--try_match_reverse", action="store_true",
                   help="Enable solving u to match reverse KL (robust scan + fallback).")
    return parser.parse_args()

def main():
    args = parse_args()
    if len(args.waymark_counts) != len(args.waymark_breakpoints) + 1:
        raise ValueError("len(waymark_counts) must equal len(breakpoints)+1")
    if len(args.epoch_values) != len(args.epoch_breakpoints) + 1:
        raise ValueError("len(epoch_values) must equal len(breakpoints)+1")

    time_id = args.time_id or strftime('%Y%m%d-%H%M', gmtime())
    generate_configs(
        target_kls=args.target_kls,
        sample_sizes=args.sample_sizes,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
        time_id=time_id,
        base_concentration=args.base_concentration,
        focus_dim=args.focus_dim,
        ratio_target=args.ratio_target,
        waymark_breakpoints=args.waymark_breakpoints,
        waymark_counts=args.waymark_counts,
        epoch_breakpoints=args.epoch_breakpoints,
        epoch_values=args.epoch_values,
        try_match_reverse=args.try_match_reverse,
        save_root=args.save_root,
        config_dir_name=args.config_dir_name,
    )


if __name__ == "__main__":
    main()