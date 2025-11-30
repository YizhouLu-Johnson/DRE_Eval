"""
Generate TDRE configs for 10D Gaussian experiments with controlled KL gaps.

This script mirrors the existing ``make_configs.py`` workflow but restricts
itself to a handy subset of settings:

- Dataset: ``gaussians_10d`` (wrapper around ``GAUSSIANS`` data handler).
- Target KL divergences: defaults to 10, 15 and 20 nats (can be overridden).
- Training sample sizes: defaults to 50, 100 and 300 samples (per trial).
- Trials: by default produce 10 independent configs per (KL, sample_size).

Running this file will drop JSON configs under
``configs/gaussians_10d/model/`` and pre-populate the save directories so
they can be passed directly to ``build_bridges.py``.
"""

import argparse
from copy import deepcopy
from time import gmtime, strftime
from typing import Iterable, List

import numpy as np

import make_configs
from make_configs import make_base_config, save_config


def _mean_shift_for_kl(target_kl: float, n_dims: int) -> float:
    """Return per-dimension mean shift that induces the desired KL."""
    return float(np.sqrt((2.0 * target_kl) / n_dims))


def _identity_matrix(n_dims: int) -> List[List[float]]:
    eye = np.eye(n_dims, dtype=np.float32)
    return eye.tolist()


def build_single_config(base_config,
                        target_kl: float,
                        sample_size: int,
                        trial_idx: int,
                        n_waymarks: int,
                        n_epochs: int,
                        seed: int):
    """Return a config dict customized for one (KL, n_samples, trial)."""
    cfg = deepcopy(base_config)
    n_dims = cfg["data"]["n_dims"]
    mean_shift = _mean_shift_for_kl(target_kl, n_dims)

    numerator_mean = [mean_shift] * n_dims
    denominator_mean = [0.0] * n_dims
    covariance = _identity_matrix(n_dims)

    cfg["data"]["dataset_name"] = "gaussians_10d"
    cfg["data"]["data_seed"] = int(seed)
    cfg["data"]["noise_dist_name"] = "gaussian"
    cfg["data"]["data_dist_name"] = "gaussian"
    cfg["data"]["n_dims"] = n_dims

    cfg["data"]["linear_combo_alphas"] = np.linspace(0.0, 1.0, n_waymarks + 1).tolist()
    cfg["data"]["initial_waymark_indices"] = list(range(n_waymarks + 1))

    cfg["data"]["noise_dist_gaussian_loc"] = 0.0
    cfg["data"]["noise_dist_gaussian_std"] = 1.0

    cfg["data"]["data_args"] = {
        "n_samples": int(sample_size),
        "n_dims": n_dims,
        "numerator_mean": numerator_mean,
        "numerator_cov": covariance,
        "denominator_mean": denominator_mean,
        "denominator_cov": covariance,
        "true_mutual_info": float(target_kl),
        "analytic_kl": float(target_kl),
        "target_kl": float(target_kl),
        "train_sample_size": int(sample_size),
        "trial_id": int(trial_idx)
    }

    cfg["optimisation"]["n_batch"] = min(sample_size, 128)
    cfg["optimisation"]["loss_function"] = "logistic"
    cfg["optimisation"]["energy_lr"] = 5e-4
    cfg["optimisation"]["n_epochs"] = int(n_epochs)
    cfg["optimisation"]["patience"] = 100
    cfg["optimisation"]["save_every_x_epochs"] = None

    cfg["architecture"]["network_type"] = "quadratic"
    cfg["architecture"]["quadratic_head_use_linear_term"] = True
    cfg["architecture"]["mlp_hidden_size"] = 256
    cfg["architecture"]["mlp_n_blocks"] = 2

    return cfg


def pick_value_from_schedule(x, breakpoints, values):
    assert len(values) == len(breakpoints) + 1, \
        "Need one more value than breakpoints (piecewise constant schedule)."
    for bp, val in zip(breakpoints, values):
        if x < bp:
            return val
    return values[-1]


def generate_configs(target_kls: Iterable[float],
                     sample_sizes: Iterable[int],
                     n_trials: int,
                     base_seed: int,
                     time_id: str,
                     waymark_breakpoints=None,
                     waymark_counts=None,
                     epoch_breakpoints=None,
                     epoch_values=None):
    """Create configs for every (KL, sample_size, trial)."""

    make_configs.time_id = time_id
    base_cfg = make_base_config()
    base_cfg["data"]["dataset_name"] = "gaussians_10d"
    base_cfg["data"]["n_dims"] = 10

    configs = []
    idx = 0
    for target_kl in target_kls:
        for sample_size in sample_sizes:
            for trial in range(n_trials):
                seed = base_seed + trial + int(target_kl * 100)
                n_waymarks = pick_value_from_schedule(sample_size,
                                                      waymark_breakpoints,
                                                      waymark_counts)
                n_epochs = pick_value_from_schedule(sample_size,
                                                    epoch_breakpoints,
                                                    epoch_values)
                cfg = build_single_config(base_cfg,
                                          target_kl=target_kl,
                                          sample_size=sample_size,
                                          trial_idx=trial,
                                          n_waymarks=int(n_waymarks),
                                          n_epochs=int(n_epochs),
                                          seed=seed)
                save_config(cfg, "model", idx)
                configs.append(cfg)
                idx += 1

    print(f"Generated {len(configs)} configs under time_id={time_id} "
          f"for dataset 'gaussians_10d'.")
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate gaussians_10d TDRE configs.")
    parser.add_argument("--target_kls", type=float, nargs="+",
                        default=[20.0],
                        help="Target analytic KL values (nats).")
    parser.add_argument("--sample_sizes", type=int, nargs="+",
                        default=[10, 100, 250, 500, 1000, 1500, 2000, 3000],
                        help="Training sample sizes per trial.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of independent configs per (KL, sample_size).")
    parser.add_argument("--waymark_breakpoints", type=int, nargs="+", default=[1000],
                        help="Sample-size thresholds where the number of waymarks changes.")
    parser.add_argument("--waymark_counts", type=int, nargs="+", default=[12, 12],
                        help="Waymark counts in each region (len = len(breakpoints)+1).")
    parser.add_argument("--epoch_breakpoints", type=int, nargs="+", default=[1000],
                        help="Sample-size thresholds where n_epochs changes.")
    parser.add_argument("--epoch_values", type=int, nargs="+", default=[400, 550],
                        help="Epoch counts in each region (len = len(breakpoints)+1).")
    parser.add_argument("--base_seed", type=int, default=464355,
                        help="Seed offset used to vary dataset sampling.")
    parser.add_argument("--time_id", type=str, default=None,
                        help="Time identifier used in config/save directories.")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.waymark_counts) != len(args.waymark_breakpoints) + 1:
        raise ValueError("len(waymark_counts) must equal len(waymark_breakpoints)+1")
    if len(args.epoch_values) != len(args.epoch_breakpoints) + 1:
        raise ValueError("len(epoch_values) must equal len(epoch_breakpoints)+1")

    time_id = args.time_id or strftime('%Y%m%d-%H%M', gmtime())
    generate_configs(target_kls=args.target_kls,
                     sample_sizes=args.sample_sizes,
                     n_trials=args.n_trials,
                     base_seed=args.base_seed,
                     time_id=time_id,
                     waymark_breakpoints=args.waymark_breakpoints,
                     waymark_counts=args.waymark_counts,
                     epoch_breakpoints=args.epoch_breakpoints,
                     epoch_values=args.epoch_values)


if __name__ == "__main__":
    main()
