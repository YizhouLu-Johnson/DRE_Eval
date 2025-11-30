"""
Train Donsker–Varadhan (DV) estimators on the gaussians_10d configs.

This script mirrors train_bdre_gaussians_10d.py but implements the DV objective
in PyTorch. It loads the TDRE config to reuse the exact same data parameters
and seeds, so DV, TDRE, and BDRE all see identical samples.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from __init__ import project_root
from data_handlers.gaussians import GAUSSIANS
from data_handlers.two_gaussians import kl_between_gaussians_with_mean


class DVNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu"):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        act_layer = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU
        }.get(activation.lower(), nn.ReLU)
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_layer())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load_tdre_config(config_path):
    cfg_path = Path(project_root, "configs", f"{config_path}.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find config at {cfg_path}")
    with cfg_path.open() as f:
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
    return gauss_dataset, n_train_samples


def train_dv(model, dataset, config, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Training (numerator) samples
    x_p1 = torch.tensor(dataset.trn.x, dtype=torch.float32, device=device)

    n_train = x_p1.shape[0]
    batch_size = min(config["batch_size"], n_train)
    batch_size = max(batch_size, 4)

    # Fix approximate number of optimizer updates across different n_train
    target_updates = int(config.get("target_updates", 800))
    n_batches = max(1, n_train // batch_size)
    n_epochs = max(1, target_updates // n_batches)

    # Early stopping based on validation DV estimate
    patience = int(config.get("patience", 30))
    patience_counter = 0
    best_state = None
    best_val_dv = -float("inf")

    # DV stabilization variables (for training objective)
    ema_Z = None
    eps = 1e-8
    ema_decay = 0.01

    # Prepare validation numerator data (fallback to train if val not available)
    if hasattr(dataset, "val") and hasattr(dataset.val, "x"):
        x_val_np = dataset.val.x
    else:
        x_val_np = dataset.trn.x
    x_val = torch.tensor(x_val_np, dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(n_batches):
            # Sample minibatch from P (numerator)
            idx_p1 = torch.randint(0, n_train, (batch_size,), device=device)
            f_p1 = model(x_p1[idx_p1])

            # Sample minibatch from Q (denominator)
            denom_np = dataset.sample_denominator(batch_size)
            x_p0 = torch.tensor(denom_np, dtype=torch.float32, device=device)
            f_p0 = model(x_p0)

            # Center critic outputs across P and Q
            f_all = torch.cat([f_p1, f_p0], dim=0)
            f_mean = f_all.mean()
            f_p1 = f_p1 - f_mean
            f_p0 = f_p0 - f_mean

            # Clip outputs to a safe range
            f_p1 = torch.clamp(f_p1, -10.0, 10.0)
            f_p0 = torch.clamp(f_p0, -10.0, 10.0)

            # DDDE-style EMA surrogate for log E_Q[e^{T}]
            exp_f0 = torch.exp(f_p0)
            batch_Z = exp_f0.mean()
            with torch.no_grad():
                if ema_Z is None:
                    ema_Z = batch_Z.detach()
                else:
                    ema_Z = (1.0 - ema_decay) * ema_Z + ema_decay * batch_Z.detach()
            Z_bar = ema_Z.detach()
            log_term = (batch_Z / (Z_bar + eps)) + torch.log(Z_bar + eps) - 1.0

            dv_obj = f_p1.mean() - log_term
            loss = -dv_obj

            optimizer.zero_grad()
            loss.backward()
            if config.get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        # ---- Validation DV estimate for early stopping ----
        model.eval()
        with torch.no_grad():
            # Subsample validation data for stability
            n_val = x_val.shape[0]
            n_eval_p1 = min(2048, n_val)
            if n_eval_p1 < n_val:
                idx_val = torch.randint(0, n_val, (n_eval_p1,), device=device)
                x_val_batch = x_val[idx_val]
            else:
                x_val_batch = x_val

            # Numerator term on validation split
            f_p1_val = model(x_val_batch)

            # Denominator samples for validation DV estimate
            # Use a somewhat larger batch for Q to stabilize log E_Q[exp(T)]
            n_eval_p0 = max(n_eval_p1 * 2, 2048)
            denom_val = dataset.sample_denominator(n_eval_p0)
            x_p0_val = torch.tensor(denom_val, dtype=torch.float32, device=device)
            f_p0_val = model(x_p0_val)

            # True DV estimate on validation:
            # E_P[f] - log E_Q[exp(f)]
            log_term_val = torch.logsumexp(f_p0_val, dim=0) - torch.log(
                torch.tensor(float(f_p0_val.shape[0]), device=device)
            )
            val_dv_est = f_p1_val.mean() - log_term_val

        val_dv_scalar = float(val_dv_est.item())

        # We choose the model with the largest validation DV lower bound
        if val_dv_scalar > best_val_dv + 1e-4:
            best_val_dv = val_dv_scalar
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Fallback: if best_state was never set, use final parameters
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return best_state


def save_metadata(save_dir, tdre_config, dv_config, true_kl):
    metadata = {
        "model_type": "dv",
        "dataset_name": tdre_config["data"]["dataset_name"],
        "data_seed": tdre_config["data"]["data_seed"],
        "data_args": tdre_config["data"]["data_args"],
        "frac": tdre_config["data"].get("frac", 1.0),
        "n_batch": tdre_config["optimisation"]["n_batch"],
        "true_kl": true_kl,
        "training_config": dv_config
    }
    metadata = _ensure_serializable(metadata)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DV estimator for gaussians_10d configs")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path like gaussians_10d/model/0")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer sizes of the DV network.")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_updates", type=int, default=800,
                        help="Approximate total number of optimizer updates per run.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Gradient norm clip value (set <=0 to disable).")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Extra offset added to TDRE data_seed.")
    parser.add_argument("--save_root", type=str, default="dv_gaussians_10d",
                        help="Subdirectory under saved_models/")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tdre_config = _load_tdre_config(args.config_path)
    data_cfg = tdre_config["data"]
    seed = int(data_cfg.get("data_seed", 0)) + args.seed_offset
    dataset, n_train_samples = build_dataset(data_cfg["data_args"], int(data_cfg["n_dims"]), seed)

    dv_config = {
        "input_dim": int(data_cfg["n_dims"]),
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "grad_clip": args.grad_clip if args.grad_clip > 0 else None,
        "n_train_samples": n_train_samples,
        "target_updates": args.target_updates
    }

    stub = _extract_stub(data_cfg["save_dir"])
    save_dir = os.path.join(project_root, "saved_models", args.save_root, stub)
    os.makedirs(save_dir, exist_ok=True)

    model = DVNet(dv_config["input_dim"], dv_config["hidden_dims"], dv_config["activation"])
    state_dict = train_dv(model, dataset, dv_config, device=torch.device(args.device))
    torch.save(state_dict, os.path.join(save_dir, "dv_model.pt"))

    true_kl = float(kl_between_gaussians_with_mean(
        np.array(data_cfg["data_args"]["numerator_mean"]),
        np.array(data_cfg["data_args"]["numerator_cov"]),
        np.array(data_cfg["data_args"]["denominator_mean"]),
        np.array(data_cfg["data_args"]["denominator_cov"])
    ))
    save_metadata(save_dir, tdre_config, dv_config, true_kl)
    print(f"Saved DV model to {save_dir}")


if __name__ == "__main__":
    main()
