import argparse
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from matplotlib import colors as mpl_colors
from train_bdre import build_bdre_graph
from train_dv_gaussians_10d import DVNet
from train_nwj_gaussians_10d import NWJNet, QuadraticNWJNet
from train_mdre_gaussians_10d import MDRENet

# TSM imports
try:
    from train_tsm_10d_tailored import (
        TimeScoreNetwork,
        compute_density_ratios,
    )
    TSM_AVAILABLE = True
except ImportError:
    TSM_AVAILABLE = False
    print("Warning: TSM evaluation not available. Install PyTorch to enable.")

tf.disable_v2_behavior()

from build_bridges import build_graph, get_feed_dict
from experiment_ops import load_model
from evaluate_gaussians_10d import (
    _collect_model_dirs,
    _get_val_data,
    _load_saved_config,
)
from utils.experiment_utils import project_root
from utils.distribution_utils import (
    get_distribution_params,
    infer_distribution_family,
    sample_distribution,
)


DEFAULT_TDRE_BASES = [
    "tre_gaussians_10d_kl10/kl10",
    "tre_gaussians_10d_kl20/kl20",
    "tre_gaussians_10d_kl30/kl30",
    "tre_gaussians_10d_kl40/kl40",
    "tre_gaussians_10d_kl50/kl50",
    "tre_gaussians_10d_kl60/kl60",
    "tre_gaussians_10d_kl70/kl70",
    "tre_gaussians_10d_kl80/kl80",
]

DEFAULT_BDRE_BASES = [
    "bdre_gaussians_10d_kl10/kl10",
    "bdre_gaussians_10d_kl20/kl20",
    "bdre_gaussians_10d_kl30/kl30",
    "bdre_gaussians_10d_kl40/kl40",
    "bdre_gaussians_10d_kl50/kl50",
    "bdre_gaussians_10d_kl60/kl60",
    "bdre_gaussians_10d_kl70/kl70",
    "bdre_gaussians_10d_kl80/kl80",
]

DEFAULT_DV_BASES = [
    "dv_gaussians_10d_kl10/kl10",
    "dv_gaussians_10d_kl20/kl20",
    "dv_gaussians_10d_kl30/kl30",
    "dv_gaussians_10d_kl40/kl40",
    "dv_gaussians_10d_kl50/kl50",
    "dv_gaussians_10d_kl60/kl60",
    "dv_gaussians_10d_kl70/kl70",
    "dv_gaussians_10d_kl80/kl80",
]

DEFAULT_NWJ_BASES = [
    "nwj_gaussians_10d_kl10/kl10",
    "nwj_gaussians_10d_kl20/kl20",
    "nwj_gaussians_10d_kl30/kl30",
    "nwj_gaussians_10d_kl40/kl40",
    "nwj_gaussians_10d_kl50/kl50",
    "nwj_gaussians_10d_kl60/kl60",
    "nwj_gaussians_10d_kl70/kl70",
    "nwj_gaussians_10d_kl80/kl80",
]

DEFAULT_MDRE_BASES = [
    "mdre_gaussians_10d_kl10/kl10",
    "mdre_gaussians_10d_kl20/kl20",
    "mdre_gaussians_10d_kl30/kl30",
    "mdre_gaussians_10d_kl40/kl40",
    "mdre_gaussians_10d_kl50/kl50",
    "mdre_gaussians_10d_kl60/kl60",
    "mdre_gaussians_10d_kl70/kl70",
    "mdre_gaussians_10d_kl80/kl80",
]

DEFAULT_TSM_BASES = [
    "tsm_gaussians_10d_kl10/kl10",
    "tsm_gaussians_10d_kl20/kl20",
    "tsm_gaussians_10d_kl30/kl30",
    "tsm_gaussians_10d_kl40/kl40",
    "tsm_gaussians_10d_kl50/kl50",
    "tsm_gaussians_10d_kl60/kl60",
    "tsm_gaussians_10d_kl70/kl70",
    "tsm_gaussians_10d_kl80/kl80",
]


def gaussian_kl(mu_a, cov_a, mu_b, cov_b):
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


def construct_eval_distribution(mu0, mu1, cov1, kl_shift):
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov1_inv = np.linalg.inv(cov1)
    direction = mu1 - mu0
    metric_norm = float(np.sqrt(direction.T @ cov1_inv @ direction))
    if metric_norm < 1e-8:
        direction = np.ones_like(mu0)
        metric_norm = float(np.sqrt(direction.T @ cov1_inv @ direction))
    dir_unit = direction / metric_norm
    shift_mag = np.sqrt(2.0 * kl_shift)
    shift_vec = dir_unit * shift_mag
    mu_star = mu1 + shift_vec
    return mu_star, cov1.copy()


def gather_model_dirs(explicit_dirs, base_list):
    dirs = []
    seen = OrderedDict()
    if explicit_dirs:
        for d in explicit_dirs:
            seen[d] = True
    if base_list:
        for base in base_list:
            if base is None:
                continue
            model_dirs = _collect_model_dirs(None, base)
            for d in model_dirs:
                seen[d] = True
    dirs = list(seen.keys())
    return dirs


def collect_method_dirs(name, explicit, bases):
    try:
        return gather_model_dirs(explicit, bases)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}. Skipping {name.upper()} models.")
        return []


def get_pstar_bundle(
    stub,
    config,
    kl_shift,
    eval_size,
    n_eval_trials,
    seed_offset,
    bundle_cache,
):
    key = (stub, kl_shift, eval_size, n_eval_trials)
    if key in bundle_cache:
        return bundle_cache[key]

    mu0 = np.asarray(config.data_args["denominator_mean"], dtype=np.float64)
    mu1 = np.asarray(config.data_args["numerator_mean"], dtype=np.float64)
    cov0 = np.asarray(config.data_args["denominator_cov"], dtype=np.float64)
    cov1 = np.asarray(config.data_args["numerator_cov"], dtype=np.float64)

    mu_star, cov_star = construct_eval_distribution(mu0, mu1, cov1, kl_shift)
    kl_p0_p1 = gaussian_kl(mu0, cov0, mu1, cov1)
    kl_p0_pstar = gaussian_kl(mu0, cov0, mu_star, cov_star)
    kl_pstar_p1 = gaussian_kl(mu_star, cov_star, mu1, cov1)
    kl_pstar_p0 = gaussian_kl(mu_star, cov_star, mu0, cov0)
    true_r = kl_pstar_p0 - kl_pstar_p1

    seed = int(config.data_seed + seed_offset + kl_shift * 1000)
    rng = np.random.RandomState(seed)
    samples = [
        rng.multivariate_normal(mu_star, cov_star, size=eval_size)
        for _ in range(n_eval_trials)
    ]

    bundle = {
        "mu_star": mu_star,
        "cov_star": cov_star,
        "kl_p0_p1": kl_p0_p1,
        "kl_p0_pstar": kl_p0_pstar,
        "kl_pstar_p1": kl_pstar_p1,
        "kl_pstar_p0": kl_pstar_p0,
        "true_r": true_r,
        "samples": samples,
    }
    bundle_cache[key] = bundle
    return bundle


def tdre_estimates(model_dir, config, val_dp, samples):
    tf.reset_default_graph()
    graph = build_graph(config)
    est_values = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_model(sess, "best", config)
        for batch in samples:
            feed_dict = get_feed_dict(
                graph, sess, val_dp, batch.astype(np.float32), config, train=False
            )
            neg_energies = sess.run(graph.neg_energies_of_data, feed_dict=feed_dict)
            log_ratios = np.sum(neg_energies, axis=1)
            est_values.append(float(np.mean(log_ratios)))
    return est_values


def bdre_estimates(model_dir, config, samples):
    tf.reset_default_graph()
    graph, _ = build_bdre_graph(config.training_config)
    saver = tf.train.Saver()
    checkpoint = os.path.join(
        project_root, "saved_models", model_dir, "bdre_model.ckpt"
    )
    if not tf.train.checkpoint_exists(checkpoint):
        raise FileNotFoundError(f"No BDRE checkpoint found at {checkpoint}")
    est_values = []
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        for batch in samples:
            feed_dict = {graph["x_p1"]: batch.astype(np.float32), graph["is_training"]: False}
            log_ratios = sess.run(graph["log_ratio_p1"], feed_dict=feed_dict)
            est_values.append(float(np.mean(log_ratios)))
    return est_values


def dv_estimates(model_dir, config, samples):
    full_dir = os.path.join(project_root, "saved_models", model_dir)
    dv_cfg = config.training_config
    model = DVNet(
        dv_cfg["input_dim"],
        dv_cfg["hidden_dims"],
        dv_cfg["activation"],
    )
    state_path = os.path.join(full_dir, "dv_model.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing DV model at {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    family = infer_distribution_family(config.data_args)
    _, denom_params = get_distribution_params(config.data_args, "denominator")
    rng = np.random.RandomState(int(config.data_seed) + 1357)
    est_values = []
    with torch.no_grad():
        for batch in samples:
            batch = batch.astype(np.float32)
            tensor = torch.from_numpy(batch)
            f_p = model(tensor)
            n_denom = max(len(batch) * 2, 2048)
            denom_samples = sample_distribution(family, denom_params, n_denom, rng).astype(np.float32)
            denom_tensor = torch.from_numpy(denom_samples)
            f_q = model(denom_tensor)
            log_term = torch.logsumexp(f_q, dim=0) - torch.log(
                torch.tensor(float(f_q.shape[0]), dtype=f_q.dtype)
            )
            est_kl = f_p.mean() - log_term
            est_values.append(float(est_kl.item()))
    return est_values


def nwj_estimates(model_dir, config, samples):
    full_dir = os.path.join(project_root, "saved_models", model_dir)
    nwj_cfg = config.training_config
    critic_type = nwj_cfg.get("critic_type", "quadratic")
    if critic_type == "quadratic":
        model = QuadraticNWJNet(nwj_cfg["input_dim"])
    else:
        model = NWJNet(
            nwj_cfg["input_dim"],
            nwj_cfg.get("hidden_dims", [256, 256]),
            nwj_cfg.get("activation", "relu"),
        )
    state_path = os.path.join(full_dir, "nwj_model.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing NWJ model at {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    family = infer_distribution_family(config.data_args)
    _, denom_params = get_distribution_params(config.data_args, "denominator")
    rng = np.random.RandomState(int(config.data_seed) + 2468)
    est_values = []
    with torch.no_grad():
        for batch in samples:
            batch = batch.astype(np.float32)
            tensor = torch.from_numpy(batch)
            f_p = model(tensor)
            n_denom = max(len(batch), 2048)
            denom_samples = sample_distribution(family, denom_params, n_denom, rng).astype(np.float32)
            denom_tensor = torch.from_numpy(denom_samples)
            f_q = model(denom_tensor)
            est_kl = f_p.mean() - torch.exp(f_q).mean() + 1.0
            est_values.append(float(est_kl.item()))
    return est_values


def mdre_estimates(model_dir, config, samples):
    full_dir = os.path.join(project_root, "saved_models", model_dir)
    mdre_cfg = config.training_config
    model = MDRENet(
        input_dim=mdre_cfg["input_dim"],
        hidden_dims=mdre_cfg["hidden_dims"],
        activation=mdre_cfg["activation"],
        num_classes=mdre_cfg["num_classes"],
    )
    state_path = os.path.join(full_dir, "mdre_model.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing MDRE model at {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    est_values = []
    with torch.no_grad():
        for batch in samples:
            tensor = torch.from_numpy(batch.astype(np.float32))
            logits = model(tensor)
            log_ratios = (logits[:, 0] - logits[:, 1]).cpu().numpy()
            est_values.append(float(np.mean(log_ratios)))
    return est_values


def tsm_estimates(model_dir, config, samples):
    """Estimate log density ratios using TSM model."""
    if not TSM_AVAILABLE:
        raise RuntimeError("TSM not available. Install PyTorch.")
    
    full_dir = os.path.join(project_root, "saved_models", model_dir)
    
    # Load TSM config
    cfg_path = os.path.join(full_dir, "config.json")
    with open(cfg_path, "r") as f:
        tsm_cfg = json.load(f)
    
    # Get model architecture params from training_config
    training_cfg = tsm_cfg.get("training_config", {})
    n_dims = int(training_cfg.get("input_dim", tsm_cfg.get("data_args", {}).get("n_dims", 10)))
    hidden_dim = int(training_cfg.get("hidden_dim", 256))
    n_layers = int(training_cfg.get("n_layers", 3))
    
    # Create model
    model = TimeScoreNetwork(n_dims, hidden_dim, n_layers)
    
    # Load checkpoint
    model_path = os.path.join(full_dir, "tsm_model_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing TSM model at {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    est_values = []
    for batch in samples:
        batch = batch.astype(np.float32)
        log_ratios, _ = compute_density_ratios(model, batch)
        # Negate: TSM integration convention gives log(p/q), we need log(q/p)
        est_values.append(float(np.mean(log_ratios)))
    return est_values


def evaluate_method_three_distribution(
    method,
    model_dirs,
    eval_size,
    n_eval_trials,
    kl_shift,
    seed_offset,
    val_cache,
    sample_cache,
):
    estimator_lookup = {
        "tdre": lambda d, cfg, val, smp: tdre_estimates(d, cfg, val["dp"], smp),
        "bdre": lambda d, cfg, val, smp: bdre_estimates(d, cfg, smp),
        "dv": lambda d, cfg, val, smp: dv_estimates(d, cfg, smp),
        "nwj": lambda d, cfg, val, smp: nwj_estimates(d, cfg, smp),
        "mdre": lambda d, cfg, val, smp: mdre_estimates(d, cfg, smp),
        "tsm": lambda d, cfg, val, smp: tsm_estimates(d, cfg, smp),
    }
    if method not in estimator_lookup:
        raise ValueError(f"Unknown method {method}")

    records = []
    for model_dir in sorted(model_dirs):
        stub = os.path.basename(model_dir)
        config = _load_saved_config(model_dir)
        entry = _get_val_data(config, val_cache, stub)
        bundle = get_pstar_bundle(
            stub,
            config,
            kl_shift,
            eval_size,
            n_eval_trials,
            seed_offset,
            sample_cache,
        )
        samples = bundle["samples"]
        est_values = estimator_lookup[method](model_dir, config, entry, samples)
        est_r = float(np.mean(est_values))
        abs_err = abs(est_r - bundle["true_r"])
        denom = max(abs(bundle["true_r"]), 1e-8)
        rel_err = abs_err / denom
        print(
            f"[{method.upper()}] {model_dir} -> KL(P0||P1)={bundle['kl_p0_p1']:.2f}, "
            f"KL(P0||P*)={bundle['kl_p0_pstar']:.2f}, True R={bundle['true_r']:.5f}, "
            f"Est R={est_r:.5f}, Rel err={rel_err:.5f}"
        )
        records.append(
            {
                "method": method,
                "model_dir": model_dir,
                "kl_p0_p1": bundle["kl_p0_p1"],
                "kl_p0_pstar": bundle["kl_p0_pstar"],
                "kl_pstar_p1": bundle["kl_pstar_p1"],
                "kl_pstar_p0": bundle["kl_pstar_p0"],
                "true_r": bundle["true_r"],
                "est_r": est_r,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "kl_shift": kl_shift,
                "n_samples": int(config.data_args["n_samples"]),
            }
        )
    return records


def plot_method_shift(records, method, kl_shift, save_path=None, save_pdf=None, error_range=None):
    """Plot single method with optional fixed error range for consistent color scale."""
    subset = [rec for rec in records if rec["kl_shift"] == kl_shift]
    if not subset:
        return
    xs = [rec["kl_p0_p1"] for rec in subset]
    ys = [rec["kl_p0_pstar"] for rec in subset]
    errors = [rec["rel_error"] for rec in subset]
    plt.figure(figsize=(5.2, 3.8))
    cmap = plt.get_cmap("coolwarm")
    # Use provided error_range for consistent color scale across methods
    if error_range:
        norm = mpl_colors.Normalize(vmin=error_range[0], vmax=error_range[1])
    else:
        norm = mpl_colors.Normalize(vmin=min(errors), vmax=max(errors))
    sc = plt.scatter(
        xs,
        ys,
        c=errors,
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors="k",
        linewidths=0.4,
    )
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1e-6, 0.04 * (x_max - x_min))
    y_pad = max(1e-6, 0.06 * (y_max - y_min))
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    for x, y, err in zip(xs, ys, errors):
        plt.annotate(
            f"{err:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.75, linewidth=0),
            clip_on=False,
        )
    plt.xlabel("KL(P0 || P1)", fontsize=10)
    plt.ylabel("KL(P0 || P*)", fontsize=10)
    plt.title(f"{method.upper()} : KL(P1||P*) = {kl_shift}", fontsize=9)
    plt.grid(alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("Relative error for R")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"Saved plot to {save_path}")
    if save_pdf:
        os.makedirs(os.path.dirname(save_pdf), exist_ok=True)
        plt.savefig(save_pdf, bbox_inches="tight")
        print(f"Saved PDF plot to {save_pdf}")
    plt.show()


def plot_all_methods_comparison(all_records, kl_shift, save_path=None, save_pdf=None):
    """
    Plot all methods together: X=KL(P0||P1), Y=relative error of R.
    Each method gets its own color.
    """
    # Method colors and markers
    method_styles = {
        "tdre": {"color": "blue", "marker": "o", "label": "TDRE"},
        "bdre": {"color": "red", "marker": "o", "label": "BDRE"},
        "dv": {"color": "green", "marker": "o", "label": "DV"},
        "nwj": {"color": "orange", "marker": "o", "label": "NWJ"},
        "mdre": {"color": "purple", "marker": "o", "label": "MDRE"},
        "tsm": {"color": "cyan", "marker": "o", "label": "TSM"},
    }
    
    plt.figure(figsize=(8, 5))
    
    for method, records in all_records.items():
        subset = [rec for rec in records if abs(rec["kl_shift"] - kl_shift) < 1e-6]
        if not subset:
            continue
        
        # Group by KL(P0||P1) and compute mean error per KL value
        from collections import defaultdict
        kl_to_errors = defaultdict(list)
        for rec in subset:
            kl_to_errors[round(rec["kl_p0_p1"], 2)].append(rec["rel_error"])
        
        xs = sorted(kl_to_errors.keys())
        ys_mean = [np.mean(kl_to_errors[x]) for x in xs]
        ys_std = [np.std(kl_to_errors[x]) for x in xs]
        
        style = method_styles.get(method, {"color": "gray", "marker": "x", "label": method.upper()})
        
        # Plot with error bars
        plt.errorbar(
            xs, ys_mean, yerr=ys_std,
            color=style["color"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.5,
            capsize=2,
            label=style["label"],
            alpha=0.8,
        )
    
    plt.xlabel("KL(P0 || P1)", fontsize=12)
    plt.ylabel("Relative Error of R", fontsize=12)
    plt.title(f"Method Comparison (KL(P1||P*) = {kl_shift})", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"Saved comparison plot to {save_path}")
    if save_pdf:
        os.makedirs(os.path.dirname(save_pdf), exist_ok=True)
        plt.savefig(save_pdf, bbox_inches="tight")
        print(f"Saved comparison PDF to {save_pdf}")
    plt.show()


def save_summary(method, records, kl_shifts, args, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append(f"{method.upper()} three-distribution evaluation summary")
    lines.append(f"KL shifts (KL(P1 || P*)): {kl_shifts}")
    lines.append(f"eval_size: {args.eval_size}")
    lines.append(f"n_eval_trials: {args.n_eval_trials}")
    lines.append("")
    for rec in records:
        lines.append(f"Model: {rec['model_dir']}")
        lines.append(
            "  KL(P0||P1)={:.4f}, KL(P0||P*)={:.4f}, KL(P*||P1)={:.4f}, KL(P*||P0)={:.4f}".format(
                rec["kl_p0_p1"],
                rec["kl_p0_pstar"],
                rec["kl_pstar_p1"],
                rec["kl_pstar_p0"],
            )
        )
        lines.append(
            "  True R={:.5f}, Est R={:.5f}, Abs error={:.5f}, Rel error={:.5f}, n_samples={}".format(
                rec["true_r"], rec["est_r"], rec["abs_error"], rec["rel_error"], rec["n_samples"]
            )
        )
        lines.append(f"  KL(P1||P*) shift used: {rec['kl_shift']}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TDRE models on three-distribution setup (P0, P1, P*)."
    )
    parser.add_argument(
        "--tdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit TDRE directories (relative to saved_models/).",
    )
    parser.add_argument(
        "--tdre_model_bases",
        nargs="+",
        default=DEFAULT_TDRE_BASES,
        help="Dataset/time_id prefixes used to collect TDRE runs.",
    )
    parser.add_argument(
        "--bdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit BDRE dirs.",
    )
    parser.add_argument(
        "--bdre_model_bases",
        nargs="+",
        default=DEFAULT_BDRE_BASES,
        help="Dataset/time_id prefixes used to collect BDRE runs.",
    )
    parser.add_argument(
        "--dv_model_dirs",
        nargs="+",
        default=None,
        help="Explicit DV dirs.",
    )
    parser.add_argument(
        "--dv_model_bases",
        nargs="+",
        default=DEFAULT_DV_BASES,
        help="Dataset/time_id prefixes used to collect DV runs.",
    )
    parser.add_argument(
        "--nwj_model_dirs",
        nargs="+",
        default=None,
        help="Explicit NWJ dirs.",
    )
    parser.add_argument(
        "--nwj_model_bases",
        nargs="+",
        default=DEFAULT_NWJ_BASES,
        help="Dataset/time_id prefixes used to collect NWJ runs.",
    )
    parser.add_argument(
        "--mdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit MDRE dirs.",
    )
    parser.add_argument(
        "--mdre_model_bases",
        nargs="+",
        default=DEFAULT_MDRE_BASES,
        help="Dataset/time_id prefixes used to collect MDRE runs.",
    )
    parser.add_argument(
        "--tsm_model_dirs",
        nargs="+",
        default=None,
        help="Explicit TSM dirs.",
    )
    parser.add_argument(
        "--tsm_model_bases",
        nargs="+",
        default=DEFAULT_TSM_BASES,
        help="Dataset/time_id prefixes used to collect TSM runs.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Number of P* samples per Monte Carlo evaluation.",
    )
    parser.add_argument(
        "--n_eval_trials",
        type=int,
        default=20,
        help="Monte Carlo trials (each with fresh P* samples).",
    )
    parser.add_argument(
        "--kl_shift",
        type=float,
        default=0.5,
        help="Target KL(P1 || P*) used to place the evaluation distribution.",
    )
    parser.add_argument(
        "--kl_shifts",
        type=float,
        nargs="+",
        default=[0.5],
        help="Optional list of KL shifts; overrides --kl_shift when provided.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=98765,
        help="Offset added to config.data_seed when sampling P*.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="results/three_dist_mc/tdre_scatter.png",
        help="PNG path for scatter plot.",
    )
    parser.add_argument(
        "--save_plot_pdf",
        type=str,
        default="results/three_dist_mc/tdre_scatter.pdf",
        help="PDF path for scatter plot.",
    )
    parser.add_argument(
        "--save_summary",
        type=str,
        default="results/three_dist_mc/summary.txt",
        help="Text summary output path.",
    )
    return parser.parse_args()


def format_output_path(base_path, method, kl_shift=None):
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    shift_part = ""
    if kl_shift is not None:
        shift_clean = str(kl_shift).replace(".", "p")
        shift_part = f"_shift{shift_clean}"
    return f"{root}_{method}{shift_part}{ext}"


def main():
    args = parse_args()
    method_dir_map = {
        "tdre": collect_method_dirs("tdre", args.tdre_model_dirs, args.tdre_model_bases),
        "bdre": collect_method_dirs("bdre", args.bdre_model_dirs, args.bdre_model_bases),
        "dv": collect_method_dirs("dv", args.dv_model_dirs, args.dv_model_bases),
        "nwj": collect_method_dirs("nwj", args.nwj_model_dirs, args.nwj_model_bases),
        "mdre": collect_method_dirs("mdre", args.mdre_model_dirs, args.mdre_model_bases),
        "tsm": collect_method_dirs("tsm", args.tsm_model_dirs, args.tsm_model_bases),
    }
    any_method = any(method_dir_map[m] for m in method_dir_map)
    if not any_method:
        raise ValueError("No model directories provided for any method.")

    kl_shift_list = args.kl_shifts if args.kl_shifts else [args.kl_shift]
    plot_base = (
        os.path.join(project_root, args.save_plot) if args.save_plot else None
    )
    plot_pdf_base = (
        os.path.join(project_root, args.save_plot_pdf) if args.save_plot_pdf else None
    )
    summary_base = (
        os.path.join(project_root, args.save_summary) if args.save_summary else None
    )
    val_cache = {}
    sample_cache = {}
    
    # First pass: collect all records from all methods
    all_method_records = {}
    for method, dirs in method_dir_map.items():
        if not dirs:
            continue
        print(f"\n=== Evaluating {method.upper()} ({len(dirs)} models) ===")
        method_records = []
        for kl_shift in kl_shift_list:
            shift_records = evaluate_method_three_distribution(
                method=method,
                model_dirs=dirs,
                eval_size=args.eval_size,
                n_eval_trials=args.n_eval_trials,
                kl_shift=kl_shift,
                seed_offset=args.seed_offset,
                val_cache=val_cache,
                sample_cache=sample_cache,
            )
            method_records.extend(shift_records)
        all_method_records[method] = method_records
    
    # Compute global error range for consistent color scale
    all_errors = []
    for method, records in all_method_records.items():
        all_errors.extend([rec["rel_error"] for rec in records])
    
    if all_errors:
        global_error_range = (min(all_errors), max(all_errors))
        print(f"\nGlobal error range: [{global_error_range[0]:.4f}, {global_error_range[1]:.4f}]")
    else:
        global_error_range = None
    
    # Second pass: plot individual methods with consistent color scale
    for method, method_records in all_method_records.items():
        if not method_records:
            continue
        for kl_shift in kl_shift_list:
            shift_records = [r for r in method_records if abs(r["kl_shift"] - kl_shift) < 1e-6]
            if shift_records:
                plot_png = format_output_path(plot_base, method, kl_shift)
                plot_pdf = format_output_path(plot_pdf_base, method, kl_shift)
                plot_method_shift(shift_records, method, kl_shift, plot_png, plot_pdf, 
                                  error_range=global_error_range)
        summary_path = format_output_path(summary_base, method, None)
        if summary_path:
            save_summary(method, method_records, kl_shift_list, args, summary_path)
    
    # Plot combined comparison for each KL shift
    for kl_shift in kl_shift_list:
        comparison_png = format_output_path(plot_base, "comparison", kl_shift)
        comparison_pdf = format_output_path(plot_pdf_base, "comparison", kl_shift)
        plot_all_methods_comparison(all_method_records, kl_shift, comparison_png, comparison_pdf)


if __name__ == "__main__":
    main()
