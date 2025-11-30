"""
Train BDRE models that mirror the gaussians_10d TDRE configs.

Usage:
    python train_bdre_gaussians_10d.py --config_path gaussians_10d/model/0

This loads the JSON config produced by make_gaussians_10d_configs.py, samples
matching numerator/denominator Gaussians, and trains a BDRE classifier with
the desired training sample size. Models are saved under
saved_models/bdre_gaussians_10d/<time_id>_<idx>/ so they can be paired with the
corresponding TDRE runs during evaluation.
"""

import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from __init__ import project_root
from train_bdre import train_bdre
from data_handlers.gaussians import GAUSSIANS
from data_handlers.two_gaussians import kl_between_gaussians_with_mean


def _load_tdre_config(config_path):
    cfg_path = os.path.join(project_root, "configs", f"{config_path}.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find config at {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _extract_stub(save_dir):
    return os.path.basename(os.path.normpath(save_dir))


def _ensure_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, list):
        return [_ensure_serializable(x) for x in data]
    if isinstance(data, dict):
        return {k: _ensure_serializable(v) for k, v in data.items()}
    return data


class _BDRESplit:
    def __init__(self, p0, p1):
        self.x_p0 = p0
        self.x_p1 = p1
        self.N_p0 = len(p0)
        self.N_p1 = len(p1)


class _BDREDataset:
    def __init__(self, trn, val, tst):
        self.trn = trn
        self.val = val
        self.tst = tst


def build_dataset(data_args, n_dims, seed):
    n_train_samples = int(data_args["n_samples"])

    gauss_dataset = GAUSSIANS(
        n_samples=n_train_samples,
        n_dims=n_dims,
        numerator_mean=data_args["numerator_mean"],
        numerator_cov=data_args["numerator_cov"],
        denominator_mean=data_args["denominator_mean"],
        denominator_cov=data_args["denominator_cov"],
        seed=seed
    )

    def make_split(split):
        n = split.x.shape[0]
        p1 = split.x
        p0 = gauss_dataset.sample_denominator(n)
        return _BDRESplit(p0, p1)

    dataset = _BDREDataset(
        trn=make_split(gauss_dataset.trn),
        val=make_split(gauss_dataset.val),
        tst=make_split(gauss_dataset.tst)
    )

    return dataset, n_train_samples


def make_bdre_config(args, n_dims, n_train_samples):
    batch_size = min(args.batch_size, n_train_samples)
    batch_size = max(batch_size, 8)
    config = {
        "input_dim": n_dims,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "reg_coef": args.reg_coef,
        "dropout_rate": args.dropout_rate,
        "n_train_samples": n_train_samples,
        "batch_size": batch_size,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "patience": args.patience,
    }
    return config


def save_metadata(save_dir, tdre_config, bdre_config, true_kl):
    metadata = {
        "model_type": "bdre",
        "dataset_name": tdre_config["data"]["dataset_name"],
        "data_seed": tdre_config["data"]["data_seed"],
        "data_args": tdre_config["data"]["data_args"],
        "frac": tdre_config["data"].get("frac", 1.0),
        "n_batch": tdre_config["optimisation"]["n_batch"],
        "true_kl": true_kl,
        "training_config": bdre_config
    }
    metadata = _ensure_serializable(metadata)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def parse_args():
    parser = ArgumentParser(
        description="Train BDRE models for gaussians_10d configs",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path like gaussians_10d/model/0")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 32],
                        help="BDRE MLP hidden sizes")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--reg_coef", type=float, default=1e-2)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100) #300
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Extra offset added to the TDRE data_seed")
    parser.add_argument("--save_root", type=str, default="bdre_gaussians_10d",
                        help="Subdirectory inside saved_models/")
    return parser.parse_args()


def main():
    args = parse_args()
    tdre_config = _load_tdre_config(args.config_path)
    data_cfg = tdre_config["data"]
    data_args = data_cfg["data_args"]
    n_dims = int(data_cfg["n_dims"])
    seed = int(data_cfg.get("data_seed", 0)) + args.seed_offset

    dataset, n_train_samples = build_dataset(
        data_args=data_args,
        n_dims=n_dims,
        seed=seed
    )

    bdre_config = make_bdre_config(args, n_dims, n_train_samples)

    stub = _extract_stub(data_cfg["save_dir"])
    save_dir = os.path.join(project_root, "saved_models", args.save_root, stub)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Training BDRE for config {args.config_path} -> {stub}")
    print(f"  Training samples per class: {n_train_samples}")
    print(f"  Saving to: {save_dir}")

    sess, graph, _ = train_bdre(bdre_config, dataset, save_dir)
    sess.close()

    true_kl = float(kl_between_gaussians_with_mean(
        np.array(data_args["numerator_mean"]),
        np.array(data_args["numerator_cov"]),
        np.array(data_args["denominator_mean"]),
        np.array(data_args["denominator_cov"])
    ))

    save_metadata(save_dir, tdre_config, bdre_config, true_kl)
    print("Done!")


if __name__ == "__main__":
    main()
