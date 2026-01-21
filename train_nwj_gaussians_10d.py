# /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code/train_nwj_gaussians_10d.py

"""
Train Nguyen–Wainwright–Jordan (NWJ) estimators on the gaussians_10d or dirichlet_10d configs.

This mirrors train_dv_gaussians_10d.py, but uses the NWJ f-divergence
variational lower bound for KL:

    KL(P || Q) = sup_{g > 0} [ E_P[log g] - E_Q[g] + 1 ].

We reuse the TDRE config to ensure NWJ, TDRE, BDRE, and DV all see
identical Gaussian data (same GAUSSIANS sampler, same data_seed).

Here we parameterize

    log g_theta(x) = t_theta(x),   g_theta(x) = exp(t_theta(x)),

with t_theta(x) = clamp(f_theta(x), [-5, 5]) coming from an MLP f_theta,
or from a quadratic critic specialised to Gaussians.
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
from data_handlers.dirichlet import DIRICHLET
from data_handlers.two_gaussians import kl_between_gaussians_with_mean
from utils.distribution_utils import (
    DIRICHLET as DIRICHLET_FAMILY,
    GAUSSIAN as GAUSSIAN_FAMILY,
    dirichlet_kl,
    infer_distribution_family,
)


class NWJNet(nn.Module):
    """
    Simple MLP critic f_theta(x). We interpret its (clamped) output as log g(x).

        t(x) = clamp(f_theta(x), [-5, 5])
        g(x) = exp(t(x))

    Then the NWJ objective is:

        E_P[t(x)] - E_Q[exp(t(x))] + 1.
    """
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
        # Returns real-valued f(x); we clamp and exponentiate outside.
        return self.net(x).squeeze(-1)


class QuadraticNWJNet(nn.Module):
    """
    Gaussian-tailored quadratic NWJ critic.

    We parameterize log g(x) as a quadratic form:
        t(x) = x^T S x + b^T x + c,
    where S is a symmetric matrix derived from a full matrix A.

    Then:
        g(x) = exp(t(x)),
    and the NWJ objective is:
        E_P[t(x)] - E_Q[exp(t(x))] + 1.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Symmetrize A for a proper quadratic form
        S = 0.5 * (self.A + self.A.t())
        # Compute x^T S x for each row x
        quad = (x @ S * x).sum(dim=-1)
        lin = x @ self.b
        return quad + lin + self.c


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
    """
    Reuse GAUSSIANS so NWJ sees the exact same numerator/denominator
    distributions as TDRE/BDRE/DV.
    """
    n_train_samples = int(data_args["n_samples"])
    family = infer_distribution_family(data_args)
    if family == GAUSSIAN_FAMILY:
        dataset = GAUSSIANS(
            n_samples=n_train_samples,
            n_dims=n_dims,
            numerator_mean=data_args["numerator_mean"],
            numerator_cov=data_args["numerator_cov"],
            denominator_mean=data_args["denominator_mean"],
            denominator_cov=data_args["denominator_cov"],
            seed=seed,
        )
    elif family == DIRICHLET_FAMILY:
        dataset = DIRICHLET(
            n_samples=n_train_samples,
            n_dims=n_dims,
            numerator_concentration=data_args["numerator_concentration"],
            denominator_concentration=data_args["denominator_concentration"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported distribution family {family}")
    return dataset, n_train_samples, family


def compute_true_kl(data_args):
    family = infer_distribution_family(data_args)
    if family == GAUSSIAN_FAMILY:
        return float(kl_between_gaussians_with_mean(
            np.array(data_args["numerator_mean"]),
            np.array(data_args["numerator_cov"]),
            np.array(data_args["denominator_mean"]),
            np.array(data_args["denominator_cov"]),
        ))
    if family == DIRICHLET_FAMILY:
        return float(dirichlet_kl(
            np.array(data_args["numerator_concentration"]),
            np.array(data_args["denominator_concentration"]),
        ))
    raise ValueError(f"Unsupported distribution family {family}")


def train_nwj(model, dataset, config, device):
    """
    Train NWJ critic by maximizing the empirical NWJ lower bound:

        E_P[log g_theta] - E_Q[g_theta] + 1,

    where we parameterize:

        f_theta(x) = model(x)
        t_theta(x) = clamp(f_theta(x), [-5, 5])
        g_theta(x) = exp(t_theta(x)).

    The empirical objective on a minibatch is:

        nwj_obj = mean_P[t] - mean_Q[exp(t)] + 1.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Training (numerator) samples P
    x_p1 = torch.tensor(dataset.trn.x, dtype=torch.float32, device=device)
    n_train = x_p1.shape[0]

    batch_size = min(config["batch_size"], n_train)
    batch_size = max(batch_size, 4)

    # Keep approx. fixed number of parameter updates across n_train
    target_updates = int(config.get("target_updates", 1000))
    n_batches = max(1, n_train // batch_size)
    n_epochs = max(1, target_updates // n_batches)

    # Early stopping based on validation NWJ estimate
    patience = int(config.get("patience", 30))
    patience_counter = 0
    best_state = None
    best_val_nwj = -float("inf")

    # Validation numerator data (fallback to train if val not available)
    if hasattr(dataset, "val") and hasattr(dataset.val, "x"):
        x_val_np = dataset.val.x
    else:
        x_val_np = dataset.trn.x
    x_val = torch.tensor(x_val_np, dtype=torch.float32, device=device)

    clamp_min, clamp_max = -7.0, 7.0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(n_batches):
            # P (numerator) minibatch
            idx_p1 = torch.randint(0, n_train, (batch_size,), device=device)
            x_p1_batch = x_p1[idx_p1]
            f_p1 = model(x_p1_batch)

            # Q (denominator) minibatch
            denom_np = dataset.sample_denominator(batch_size)
            x_p0 = torch.tensor(denom_np, dtype=torch.float32, device=device)
            f_p0 = model(x_p0)

            # t(x) = clamp(f(x)), g(x) = exp(t(x))
            t_p1 = torch.clamp(f_p1, clamp_min, clamp_max)
            t_p0 = torch.clamp(f_p0, clamp_min, clamp_max)

            # t_p1 = f_p1
            # t_p0 = f_p0

            g_p1 = torch.exp(t_p1)
            g_p0 = torch.exp(t_p0)

            # log g_p1 = t_p1
            log_g_p1 = t_p1

            # NWJ lower bound on this minibatch
            nwj_obj = log_g_p1.mean() - g_p0.mean() + 1.0
            loss = -nwj_obj  # maximize NWJ bound

            optimizer.zero_grad()
            loss.backward()
            if config.get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        # ---- Validation NWJ estimate for early stopping ----
        model.eval()
        with torch.no_grad():
            n_val = x_val.shape[0]
            n_eval_p1 = min(2048, n_val)
            if n_eval_p1 < n_val:
                idx_val = torch.randint(0, n_val, (n_eval_p1,), device=device)
                x_val_batch = x_val[idx_val]
            else:
                x_val_batch = x_val

            f_p1_val = model(x_val_batch)

            # Denominator samples for Q in validation
            n_eval_p0 = max(n_eval_p1 * 2, 2048)
            denom_val = dataset.sample_denominator(n_eval_p0)
            x_p0_val = torch.tensor(denom_val, dtype=torch.float32, device=device)
            f_p0_val = model(x_p0_val)

            t_p1_val = torch.clamp(f_p1_val, clamp_min, clamp_max)
            t_p0_val = torch.clamp(f_p0_val, clamp_min, clamp_max)

            # t_p1_val = f_p1_val
            # t_p0_val = f_p0_val

            g_p1_val = torch.exp(t_p1_val)
            g_p0_val = torch.exp(t_p0_val)

            log_g_p1_val = t_p1_val

            val_nwj_est = log_g_p1_val.mean() - g_p0_val.mean() + 1.0

        val_nwj_scalar = float(val_nwj_est.item())

        if val_nwj_scalar > best_val_nwj + 1e-4:
            best_val_nwj = val_nwj_scalar
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Fallback if no improvement ever recorded
    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    return best_state


def save_metadata(save_dir, tdre_config, nwj_config, true_kl):
    metadata = {
        "model_type": "nwj",
        "dataset_name": tdre_config["data"]["dataset_name"],
        "data_seed": tdre_config["data"]["data_seed"],
        "data_args": tdre_config["data"]["data_args"],
        "frac": tdre_config["data"].get("frac", 1.0),
        "n_batch": tdre_config["optimisation"]["n_batch"],
        "true_kl": true_kl,
        "training_config": nwj_config,
    }
    metadata = _ensure_serializable(metadata)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train NWJ estimator for gaussians_10d configs")
#     parser.add_argument("--config_path", type=str, required=True,
#                         help="Path like gaussians_10d/model/0")
#     parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
#                         help="Hidden layer sizes of the NWJ network.")
#     parser.add_argument("--activation", type=str, default="relu")
#     parser.add_argument("--lr", type=float, default=5e-5)
#     parser.add_argument("--weight_decay", type=float, default=5e-5,
#                         help="L2 weight decay for Adam.")
#     parser.add_argument("--n_epochs", type=int, default=500,
#                         help="[Deprecated] kept for CLI compat; effective training is via target_updates.")
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--target_updates", type=int, default=1000,
#                         help="Approximate total number of optimizer updates per run.")
#     parser.add_argument("--patience", type=int, default=30)
#     parser.add_argument("--grad_clip", type=float, default=5.0,
#                         help="Gradient norm clip value (set <=0 to disable).")
#     parser.add_argument("--seed_offset", type=int, default=0,
#                         help="Extra offset added to TDRE data_seed.")
#     parser.add_argument("--save_root", type=str, default="nwj_gaussians_10d_kl10",
#                         help="Subdirectory under saved_models/")
#     parser.add_argument("--device", type=str, default="cpu")
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="Train NWJ estimator for gaussians_10d configs")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path like gaussians_10d/model/0")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes of the NWJ network.")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay for Adam.")
    parser.add_argument("--n_epochs", type=int, default=500,
                        help="[Deprecated] kept for CLI compat; effective training is via target_updates.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_updates", type=int, default=1000,
                        help="Approximate total number of optimizer updates per run.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=3.0,
                        help="Gradient norm clip value (set <=0 to disable).")
    parser.add_argument("--critic_type", type=str, choices=["mlp", "quadratic"], default=None,
                        help="Critic architecture (defaults to quadratic for Gaussians, MLP otherwise).")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Extra offset added to TDRE data_seed.")
    parser.add_argument("--save_root", type=str, default="nwj_pstar_p0_kl10",
                        help="Subdirectory under saved_models/")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tdre_config = _load_tdre_config(args.config_path)
    data_cfg = tdre_config["data"]
    seed = int(data_cfg.get("data_seed", 0)) + args.seed_offset

    dataset, n_train_samples, family = build_dataset(
        data_cfg["data_args"],
        int(data_cfg["n_dims"]),
        seed,
    )
    critic_type = args.critic_type or ("quadratic" if family == GAUSSIAN_FAMILY else "mlp")

    nwj_config = {
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
        "target_updates": args.target_updates,
        "critic_type": critic_type,
    }

    stub = _extract_stub(data_cfg["save_dir"])
    save_dir = os.path.join(project_root, "saved_models", args.save_root, stub)
    os.makedirs(save_dir, exist_ok=True)

    # Choose which critic architecture to use for NWJ.
    # Default: generic MLP critic.
    # model = NWJNet(
    #     nwj_config["input_dim"],
    #     nwj_config["hidden_dims"],
    #     nwj_config["activation"],
    # )

    if critic_type == "quadratic":
        model = QuadraticNWJNet(nwj_config["input_dim"])
    else:
        model = NWJNet(
            nwj_config["input_dim"],
            nwj_config["hidden_dims"],
            nwj_config["activation"],
        )

    state_dict = train_nwj(model, dataset, nwj_config, device=torch.device(args.device))
    torch.save(state_dict, os.path.join(save_dir, "nwj_model.pt"))

    true_kl = compute_true_kl(data_cfg["data_args"])
    save_metadata(save_dir, tdre_config, nwj_config, true_kl)
    print(f"Saved NWJ model to {save_dir}")


if __name__ == "__main__":
    main()
