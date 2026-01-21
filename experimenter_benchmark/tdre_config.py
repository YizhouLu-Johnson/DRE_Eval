from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

import make_configs
from make_configs import make_base_config, update_config
from utils.misc_utils import AttrDict, merge_dicts


@dataclass
class TDREConfigInputs:
    dim: int
    num_samples: int
    n_waymarks: int
    n_epochs: int
    patience: int
    batch_size: int
    energy_lr: float
    loss_function: str
    optimizer: str
    loss_decay_factor: float
    network_type: str
    quadratic_constraint_type: str
    quadratic_use_linear_term: bool
    mlp_hidden_size: int
    mlp_n_blocks: int
    waymark_mechanism: str
    shuffle_waymarks: bool
    save_root: Optional[Path]


def _gaussian_kl_same_mean(cov_p: np.ndarray, cov_q: np.ndarray) -> float:
    dim = cov_p.shape[0]
    inv_q = np.linalg.inv(cov_q)
    trace_term = np.trace(inv_q @ cov_p)
    log_det = np.linalg.slogdet(cov_q)[1] - np.linalg.slogdet(cov_p)[1]
    return 0.5 * (trace_term - dim + log_det)


def build_tdre_config(
    inputs: TDREConfigInputs,
    joint_cov: np.ndarray,
    product_cov: np.ndarray,
    rng_seed: int,
    config_dir_name: str,
    save_dir_root: str,
    time_id: str,
    config_id: str,
) -> AttrDict:
    base_config = make_base_config()
    waymark_indices = list(range(inputs.n_waymarks + 1))
    alphas = np.linspace(0.0, 1.0, inputs.n_waymarks + 1).tolist()
    analytic_kl = _gaussian_kl_same_mean(joint_cov, product_cov)
    centered_mean = np.zeros(inputs.dim)

    base_config["data"].update({
        "dataset_name": "gaussians",
        "data_seed": int(rng_seed),
        "noise_dist_name": "gaussian",
        "data_dist_name": "gaussian",
        "n_dims": inputs.dim,
        "linear_combo_alphas": alphas,
        "initial_waymark_indices": waymark_indices,
        "waymark_mechanism": inputs.waymark_mechanism,
        "shuffle_waymarks": inputs.shuffle_waymarks,
        "noise_dist_gaussian_loc": 0.0,
        "noise_dist_gaussian_std": 1.0,
        "config_dir_name": config_dir_name,
        "save_dir_root": save_dir_root,
        "data_args": {
            "n_samples": int(inputs.num_samples),
            "n_dims": inputs.dim,
            "numerator_mean": centered_mean.tolist(),
            "numerator_cov": joint_cov.tolist(),
            "denominator_mean": centered_mean.tolist(),
            "denominator_cov": product_cov.tolist(),
            "true_mutual_info": float(analytic_kl),
            "analytic_kl": float(analytic_kl),
            "target_kl": float(analytic_kl),
            "train_sample_size": int(inputs.num_samples),
        },
    })

    base_config["optimisation"].update({
        "loss_function": inputs.loss_function,
        "optimizer": inputs.optimizer,
        "n_epochs": int(inputs.n_epochs),
        "n_batch": int(min(inputs.num_samples, inputs.batch_size)),
        "energy_lr": float(inputs.energy_lr),
        "patience": int(inputs.patience),
        "num_losses": inputs.n_waymarks,
        "loss_decay_factor": float(inputs.loss_decay_factor),
    })

    base_config["architecture"].update({
        "network_type": inputs.network_type,
        "quadratic_constraint_type": inputs.quadratic_constraint_type,
        "quadratic_head_use_linear_term": inputs.quadratic_use_linear_term,
        "mlp_hidden_size": inputs.mlp_hidden_size,
        "mlp_n_blocks": inputs.mlp_n_blocks,
    })

    make_configs.time_id = time_id
    update_config(base_config, 0)

    merged = merge_dicts(
        base_config["data"],
        base_config["architecture"],
        base_config["optimisation"],
        base_config["ais"],
        base_config["flow"],
    )
    merged["cov_mat"] = product_cov.astype(np.float32)
    merged["epoch_idx"] = -1
    merged["config_id"] = config_id

    base_config["data"]["config_id"] = config_id

    return AttrDict(base_config), AttrDict(merged)


def write_tdre_config(
    config: AttrDict,
    config_path: Path,
) -> None:
    def _jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {key: _jsonify(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(value) for value in obj]
        return obj

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        import json
        json.dump(_jsonify(dict(config)), handle, indent=4)


def write_tdre_script(
    config_paths: Iterable[Path],
    script_path: Path,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for path in config_paths:
        lines.append(f"python build_bridges.py --config_path={path.as_posix()}")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def append_tdre_script_line(
    script_path: Path,
    config_path: Path,
    config_ref: Optional[str] = None,
) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    if not script_path.exists():
        script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
        script_path.chmod(0o755)
    with script_path.open("a", encoding="utf-8") as handle:
        ref = config_ref if config_ref is not None else config_path.as_posix()
        handle.write(f"python build_bridges.py --config_path={ref}\n")
