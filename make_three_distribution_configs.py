#!/usr/bin/env python3
"""
Create three-distribution configs from base Dirichlet configs.

Given base configs (P1 numerator, P0 denominator), construct P* such that:
  - KL(P* || P1) = pstar_kl  (exact)
and write:
  - local configs:   numerator=P*, denominator=P1
  - distant configs: numerator=P*, denominator=P0

Construction of P*:
  alpha_star = scale_to_sum(alpha1 * exp(t w)), with balanced direction w,
  solve t so KL(alpha_star || alpha1) = pstar_kl.

With the base generator tuned so KL(P1||P0) tracks KL(P0||P1),
KL(P*||P0) will naturally grow with sweep and stay “not too far”.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gammaln, psi

from __init__ import project_root


# ---------------------------
# Dirichlet KL
# ---------------------------

def dirichlet_kl(alpha_p, alpha_q) -> float:
    alpha_p = np.asarray(alpha_p, dtype=np.float64)
    alpha_q = np.asarray(alpha_q, dtype=np.float64)
    term1 = np.sum(gammaln(alpha_q)) - gammaln(np.sum(alpha_q))
    term2 = -np.sum(gammaln(alpha_p)) + gammaln(np.sum(alpha_p))
    term3 = np.sum((alpha_p - alpha_q) * (psi(alpha_p) - psi(np.sum(alpha_p))))
    return float(term1 + term2 + term3)


# ---------------------------
# P* construction
# ---------------------------

def random_balanced_direction(d: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.normal(size=d).astype(np.float64)
    w = w - np.mean(w)  # sum=0
    n = np.linalg.norm(w)
    if n < 1e-12:
        w = np.ones(d, dtype=np.float64)
        w = w - np.mean(w)
        n = np.linalg.norm(w)
    return w / n

def scale_to_sum(alpha: np.ndarray, target_sum: float) -> np.ndarray:
    s = float(np.sum(alpha))
    return alpha * (target_sum / s)

def make_alpha_star(alpha1: np.ndarray, w: np.ndarray, t: float) -> np.ndarray:
    # exp-move with sum preserved (keeps concentrations comparable)
    raw = alpha1 * np.exp(t * w)
    return scale_to_sum(raw, float(np.sum(alpha1)))

def solve_t_for_kl_star_p1(alpha1: np.ndarray, w: np.ndarray, target_kl: float) -> float:
    def f(t: float) -> float:
        a_star = make_alpha_star(alpha1, w, t)
        return dirichlet_kl(a_star, alpha1) - target_kl

    lo, hi = 0.0, 0.05
    while f(hi) < 0:
        hi *= 2.0
        if hi > 200:
            raise RuntimeError("Could not bracket t for KL(P*||P1).")
    sol = root_scalar(f, bracket=[lo, hi], method="brentq", xtol=1e-12)
    if not sol.converged:
        raise RuntimeError("t solve for KL(P*||P1) failed.")
    return float(sol.root)

def construct_pstar(alpha1: np.ndarray, pstar_kl: float, seed: int) -> (np.ndarray, dict):
    rng = np.random.default_rng(int(seed))
    d = len(alpha1)
    w = random_balanced_direction(d, rng)
    t = solve_t_for_kl_star_p1(alpha1, w, pstar_kl)
    alpha_star = make_alpha_star(alpha1, w, t)
    return alpha_star, {"w": w.tolist(), "t": float(t)}


# ---------------------------
# IO helpers
# ---------------------------

def _prepare_paths(save_root, time_id, stub):
    save_dir = Path(project_root) / "saved_models" / save_root / f"{time_id}_{stub}"
    fig_dir = Path(project_root) / "figs" / save_root / f"{time_id}/"
    fig_dir = fig_dir.with_suffix("")
    return str(save_dir), str(fig_dir)

def _write_config(cfg, out_dir, idx):
    out_dir = Path(project_root) / "configs" / out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{idx}.json").open("w") as f:
        json.dump(cfg, f, indent=2)

def _resolve_with_kl(base, kl_value):
    label = f"kl{int(round(float(kl_value)))}"
    return str(Path(base) / label)


# ---------------------------
# Main builder
# ---------------------------

def build_configs(base_dir: str,
                  local_dir: str,
                  distant_dir: str,
                  local_save_root: str,
                  distant_save_root: str,
                  time_id_local: str,
                  time_id_distant: str,
                  pstar_kl: float):

    base_paths = sorted(Path(base_dir).glob("*.json"))
    if not base_paths:
        raise ValueError(f"No base configs found in {base_dir}")

    local_idx = 0
    distant_idx = 0

    for path in base_paths:
        cfg = json.load(path.open())
        args = cfg["data"]["data_args"]

        # base convention: numerator=P1, denominator=P0
        alpha1 = np.array(args["numerator_concentration"], dtype=np.float64)
        alpha0 = np.array(args["denominator_concentration"], dtype=np.float64)

        # Prefer sweep KL(P0||P1) stored by base generator
        base_kl_sweep = float(args.get("base_kl_p0_p1", dirichlet_kl(alpha0, alpha1)))

        # Build P* close to P1
        seed = int(cfg["data"].get("data_seed", 0)) + 99991
        alpha_star, pstar_meta = construct_pstar(alpha1, pstar_kl=pstar_kl, seed=seed)

        local_dir_with_kl = _resolve_with_kl(local_dir, base_kl_sweep)
        distant_dir_with_kl = _resolve_with_kl(distant_dir, base_kl_sweep)
        local_save_root_with_kl = _resolve_with_kl(local_save_root, base_kl_sweep)
        distant_save_root_with_kl = _resolve_with_kl(distant_save_root, base_kl_sweep)

        # -------- local: (P*, P1)
        local_cfg = deepcopy(cfg)
        local_args = local_cfg["data"]["data_args"]

        local_args["numerator_concentration"] = alpha_star.tolist()
        local_args["denominator_concentration"] = alpha1.tolist()

        kl_star_p1 = dirichlet_kl(alpha_star, alpha1)
        local_args["analytic_kl"] = float(kl_star_p1)
        local_args["target_kl"] = float(kl_star_p1)
        local_args["true_mutual_info"] = float(kl_star_p1)

        local_args["pstar_kl_target"] = float(pstar_kl)
        local_args["pstar_move_t"] = float(pstar_meta["t"])
        local_args["pstar_move_dir"] = pstar_meta["w"]
        local_args["base_kl_sweep_p0_p1"] = float(base_kl_sweep)

        local_cfg["data"]["save_dir_root"] = local_save_root_with_kl
        local_cfg["data"]["config_dir_name"] = local_dir_with_kl
        local_cfg["data"]["noise_dist_name"] = "dirichlet"
        local_cfg["data"]["noise_dist_dirichlet_concentration"] = alpha1.tolist()

        save_dir, fig_dir = _prepare_paths(local_save_root_with_kl, time_id_local, local_idx)
        local_cfg["data"]["save_dir"] = save_dir
        local_cfg["data"]["fig_dir_name"] = fig_dir
        _write_config(local_cfg, local_dir_with_kl, local_idx)
        local_idx += 1

        # -------- distant: (P*, P0)
        distant_cfg = deepcopy(cfg)
        distant_args = distant_cfg["data"]["data_args"]

        distant_args["numerator_concentration"] = alpha_star.tolist()
        distant_args["denominator_concentration"] = alpha0.tolist()

        kl_star_p0 = dirichlet_kl(alpha_star, alpha0)
        distant_args["analytic_kl"] = float(kl_star_p0)
        distant_args["target_kl"] = float(kl_star_p0)
        distant_args["true_mutual_info"] = float(kl_star_p0)

        distant_args["pstar_kl_target"] = float(pstar_kl)
        distant_args["pstar_move_t"] = float(pstar_meta["t"])
        distant_args["pstar_move_dir"] = pstar_meta["w"]
        distant_args["base_kl_sweep_p0_p1"] = float(base_kl_sweep)

        distant_cfg["data"]["save_dir_root"] = distant_save_root_with_kl
        distant_cfg["data"]["config_dir_name"] = distant_dir_with_kl
        distant_cfg["data"]["noise_dist_name"] = "dirichlet"
        distant_cfg["data"]["noise_dist_dirichlet_concentration"] = alpha0.tolist()

        save_dir, fig_dir = _prepare_paths(distant_save_root_with_kl, time_id_distant, distant_idx)
        distant_cfg["data"]["save_dir"] = save_dir
        distant_cfg["data"]["fig_dir_name"] = fig_dir
        _write_config(distant_cfg, distant_dir_with_kl, distant_idx)
        distant_idx += 1

    print(f"Wrote {local_idx} local configs under configs/{local_dir}/kl*/model")
    print(f"Wrote {distant_idx} distant configs under configs/{distant_dir}/kl*/model")


def parse_args():
    p = argparse.ArgumentParser("Make three-distribution Dirichlet configs from base configs.")
    p.add_argument("--base_config_dir", type=str,
                   default=str(Path(project_root) / "configs/dirichlet_10d_80/model"))
    p.add_argument("--local_config_dir", type=str, default="dirichlet_10d_pstar_p1_kl80")
    p.add_argument("--distant_config_dir", type=str, default="dirichlet_10d_pstar_p0_kl80")
    p.add_argument("--local_save_root", type=str, default="dirichlet_tdre_pstar_p1_kl80")
    p.add_argument("--distant_save_root", type=str, default="dirichlet_tdre_pstar_p0_kl80")
    p.add_argument("--time_id_local", type=str, default="kl80")
    p.add_argument("--time_id_distant", type=str, default="kl80")
    p.add_argument("--pstar_kl", type=float, default=0.5)

    # If you want KL(P*||P0) in [base_kl_p0_p1 - bw, base_kl_p0_p1 + bw]
    p.add_argument("--pstar_p0_bandwidth", type=float, default=10.0,
                   help="Set to e.g. 10 for [K-10, K+10]. Set to -1 to disable.")
    return p.parse_args()


def main():
    args = parse_args()

    build_configs(
        base_dir=args.base_config_dir,
        local_dir=args.local_config_dir,
        distant_dir=args.distant_config_dir,
        local_save_root=args.local_save_root,
        distant_save_root=args.distant_save_root,
        time_id_local=args.time_id_local,
        time_id_distant=args.time_id_distant,
        pstar_kl=args.pstar_kl,
    )


if __name__ == "__main__":
    main()
