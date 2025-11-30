"""
Train MDRE (multinomial logistic regression density ratio estimator) on the gaussians_10d configs.

This script mirrors the TDRE/BDRE/DV training wrappers: it loads the TDRE JSON config to reuse
the exact same Gaussian parameters/seed, generates the same number of samples per distribution,
and trains a multi-class classifier across {p, q, auxiliary waymarks}. The learned logits h_c(x)
allow us to recover log p/q = h_p(x) - h_q(x).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from __init__ import project_root
from data_handlers.gaussians import GAUSSIANS
from data_handlers.two_gaussians import kl_between_gaussians_with_mean


class MDRENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str, num_classes: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        act_layer = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }.get(activation.lower(), nn.ReLU)

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_layer())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_tdre_config(config_path: str):
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
    gauss_dataset = GAUSSIANS(
        n_samples=int(data_args["n_samples"]),
        n_dims=n_dims,
        numerator_mean=data_args["numerator_mean"],
        numerator_cov=data_args["numerator_cov"],
        denominator_mean=data_args["denominator_mean"],
        denominator_cov=data_args["denominator_cov"],
        seed=seed,
    )
    return gauss_dataset


def make_aux_samples(alpha, p_samples, q_samples):
    n = min(len(p_samples), len(q_samples))
    return (1.0 - alpha) * p_samples[:n] + alpha * q_samples[:n]


def prepare_class_samples(dataset, aux_alphas, n_train, rng):
    """Return list of arrays for classes [p, q, aux...]"""
    samples = []
    # Numerator: reuse trn.x (already size n_train)
    if dataset.trn.x.shape[0] >= n_train:
        samples.append(dataset.trn.x[:n_train])
    else:
        idx = rng.randint(dataset.trn.x.shape[0], size=n_train)
        samples.append(dataset.trn.x[idx])

    # Denominator: sample once
    samples.append(dataset.sample_denominator(n_train))

    # Aux distributions via linear mixing of numerator/denominator draws
    for alpha in aux_alphas:
        p_draws = dataset.sample_data(n_train)
        q_draws = dataset.sample_denominator(n_train)
        aux = (1.0 - alpha) * p_draws + alpha * q_draws
        samples.append(aux)

    return samples


def build_dataloader(class_samples, batch_size, device):
    xs = np.concatenate(class_samples, axis=0)
    labels = []
    for cls_idx, arr in enumerate(class_samples):
        labels.extend([cls_idx] * len(arr))
    ys = np.array(labels, dtype=np.int64)

    dataset = TensorDataset(
        torch.tensor(xs, dtype=torch.float32, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_mdre(model, train_loader, val_data, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    best_state = None
    best_val_loss = float("inf")
    patience_cnt = 0

    for epoch in range(config["n_epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits = model(val_data["x"])
            loss = criterion(logits, val_data["y"])
        if loss.item() < best_val_loss - 1e-5:
            best_val_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= config["patience"]:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return best_state


def build_val_data(dataset, aux_alphas, n_val, device):
    if dataset.val.x.shape[0] >= n_val:
        p_val = dataset.val.x[:n_val]
    else:
        idx = np.random.randint(dataset.val.x.shape[0], size=n_val)
        p_val = dataset.val.x[idx]

    q_val = dataset.sample_denominator(n_val)

    xs = [p_val, q_val]
    labels = [np.zeros(len(p_val), dtype=np.int64), np.ones(len(q_val), dtype=np.int64)]

    for j, alpha in enumerate(aux_alphas):
        aux = (1.0 - alpha) * dataset.sample_data(n_val) + alpha * dataset.sample_denominator(n_val)
        xs.append(aux)
        labels.append(np.full(len(aux), 2 + j, dtype=np.int64))

    x_tensor = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.concatenate(labels, axis=0), dtype=torch.long, device=device)
    return {"x": x_tensor, "y": y_tensor}


def save_metadata(save_dir, tdre_config, mdre_config, true_kl):
    metadata = {
        "model_type": "mdre",
        "dataset_name": tdre_config["data"]["dataset_name"],
        "data_seed": tdre_config["data"]["data_seed"],
        "data_args": tdre_config["data"]["data_args"],
        "frac": tdre_config["data"].get("frac", 1.0),
        "n_batch": tdre_config["optimisation"]["n_batch"],
        "true_kl": true_kl,
        "training_config": mdre_config,
    }
    metadata = _ensure_serializable(metadata)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDRE estimator for gaussians_10d configs")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path like gaussians_10d/model/0")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes.")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Extra offset added to TDRE data_seed.")
    parser.add_argument("--save_root", type=str, default="mdre_gaussians_10d",
                        help="Subdirectory under saved_models/")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tdre_config = _load_tdre_config(args.config_path)
    data_cfg = tdre_config["data"]
    seed = int(data_cfg.get("data_seed", 0)) + args.seed_offset
    dataset = build_dataset(data_cfg["data_args"], int(data_cfg["n_dims"]), seed)

    aux_alphas_full = tdre_config["data"].get("linear_combo_alphas", [])
    aux_alphas = aux_alphas_full[1:-1] if len(aux_alphas_full) > 2 else [0.5]

    rng = np.random.RandomState(seed)
    n_train = int(data_cfg["data_args"]["n_samples"])
    class_samples = prepare_class_samples(dataset, aux_alphas, n_train, rng)
    train_loader = build_dataloader(class_samples, args.batch_size, torch.device(args.device))

    val_data = build_val_data(dataset, aux_alphas, min(1024, n_train), torch.device(args.device))

    num_classes = 2 + len(aux_alphas)
    model = MDRENet(
        input_dim=int(data_cfg["n_dims"]),
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        num_classes=num_classes,
    )

    mdre_config = {
        "input_dim": int(data_cfg["n_dims"]),
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "grad_clip": args.grad_clip if args.grad_clip > 0 else None,
        "auxiliary_alphas": aux_alphas,
        "num_classes": num_classes,
        "n_train_samples": n_train,
    }

    state_dict = train_mdre(model, train_loader, val_data, mdre_config, torch.device(args.device))

    stub = _extract_stub(data_cfg["save_dir"])
    save_dir = os.path.join(project_root, "saved_models", args.save_root, stub)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(save_dir, "mdre_model.pt"))

    true_kl = float(
        kl_between_gaussians_with_mean(
            np.array(data_cfg["data_args"]["numerator_mean"]),
            np.array(data_cfg["data_args"]["numerator_cov"]),
            np.array(data_cfg["data_args"]["denominator_mean"]),
            np.array(data_cfg["data_args"]["denominator_cov"]),
        )
    )
    save_metadata(save_dir, tdre_config, mdre_config, true_kl)
    print(f"Saved MDRE model to {save_dir}")


if __name__ == "__main__":
    main()
