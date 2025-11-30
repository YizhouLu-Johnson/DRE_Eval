import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from scipy.special import logsumexp

tf.disable_v2_behavior()

from build_bridges import build_graph, get_feed_dict
from experiment_ops import load_model
from train_bdre import build_bdre_graph
from train_dv_gaussians_10d import DVNet
from train_nwj_gaussians_10d import NWJNet, QuadraticNWJNet
from train_mdre_gaussians_10d import MDRENet
from utils.experiment_utils import project_root, load_data_providers_and_update_conf
from utils.misc_utils import AttrDict


def _load_saved_config(model_dir):
    cfg_path = os.path.join(project_root, "saved_models", model_dir, "config.json")
    with open(cfg_path, "r") as f:
        return AttrDict(json.load(f))


def _collect_model_dirs(explicit_dirs, base):
    if explicit_dirs:
        return explicit_dirs
    if not base:
        return []

    dataset, base_id = base.split("/", 1)
    dataset_dir = Path(project_root) / "saved_models" / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Could not find directory {dataset_dir}")

    dirs = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and child.name.startswith(f"{base_id}_"):
            dirs.append(f"{dataset}/{child.name}")

    if not dirs:
        raise ValueError(f"No saved models found under prefix {base}")

    return dirs


def _true_kl(config):
    data_args = config.data_args
    return float(
        data_args.get(
            "analytic_kl",
            data_args.get("target_kl", data_args.get("true_mutual_info")),
        )
    )


def _get_val_data(config, cache, key):
    if key in cache:
        return cache[key]
    train_dp, val_dp = load_data_providers_and_update_conf(config, shuffle=False)
    entry = {
        "data": val_dp.data.astype(np.float32),
        "dp": val_dp,
        "config": config,
        "source": train_dp.source,
    }
    cache[key] = entry
    return entry


def evaluate_tdre_models(model_dirs, eval_size, n_eval_trials, val_cache):
    grouped = defaultdict(list)
    per_model = []

    for model_dir in sorted(model_dirs):
        stub = os.path.basename(model_dir)
        config = _load_saved_config(model_dir)
        entry = _get_val_data(config, val_cache, stub)
        val_data = entry["data"]
        val_dp = entry["dp"]

        tf.reset_default_graph()
        graph = build_graph(config)
        errors = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            load_model(sess, "best", config)

            for _ in range(n_eval_trials):
                n_eval = min(eval_size, val_data.shape[0])
                idx = np.random.choice(val_data.shape[0], size=n_eval, replace=False)
                batch = val_data[idx]

                feed_dict = get_feed_dict(
                    graph, sess, val_dp, batch, config, train=False
                )
                neg_energies = sess.run(graph.neg_energies_of_data, feed_dict=feed_dict)
                log_ratios = np.sum(neg_energies, axis=1)
                est_kl = float(np.mean(log_ratios))
                rel_error = abs(est_kl - _true_kl(config)) / _true_kl(config)
                errors.append(rel_error)

        avg_error = float(np.mean(errors))
        grouped[(_true_kl(config), int(config.data_args["n_samples"]))].append(
            avg_error
        )
        per_model.append(
            (
                model_dir,
                _true_kl(config),
                int(config.data_args["n_samples"]),
                avg_error,
            )
        )

    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))

    return per_model, summary, val_cache


def evaluate_bdre_models(model_dirs, eval_size, n_eval_trials, val_cache):
    grouped = defaultdict(list)
    per_model = []

    for model_dir in sorted(model_dirs):
        stub = os.path.basename(model_dir)
        config = _load_saved_config(model_dir)

        if stub in val_cache:
            entry = val_cache[stub]
        else:
            entry = _get_val_data(config, val_cache, stub)
        val_data = entry["data"]

        tf.reset_default_graph()
        graph, _ = build_bdre_graph(config.training_config)
        saver = tf.train.Saver()
        checkpoint = os.path.join(
            project_root, "saved_models", model_dir, "bdre_model.ckpt"
        )
        if not tf.train.checkpoint_exists(checkpoint):
            raise FileNotFoundError(f"No BDRE checkpoint found at {checkpoint}")

        errors = []
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            for _ in range(n_eval_trials):
                n_eval = min(eval_size, val_data.shape[0])
                idx = np.random.choice(val_data.shape[0], size=n_eval, replace=False)
                batch = val_data[idx]
                feed_dict = {graph["x_p1"]: batch, graph["is_training"]: False}
                log_ratios = sess.run(graph["log_ratio_p1"], feed_dict=feed_dict)
                est_kl = float(np.mean(log_ratios))
                rel_error = abs(est_kl - config.true_kl) / abs(config.true_kl)
                errors.append(rel_error)

        avg_error = float(np.mean(errors))
        grouped[(config.true_kl, int(config.data_args["n_samples"]))].append(avg_error)
        per_model.append(
            (model_dir, config.true_kl, int(config.data_args["n_samples"]), avg_error)
        )

    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))

    return per_model, summary, val_cache


def evaluate_dv_models(model_dirs, eval_size, n_eval_trials, val_cache):
    grouped = defaultdict(list)
    per_model = []

    for model_dir in sorted(model_dirs):
        full_dir = os.path.join(project_root, "saved_models", model_dir)
        config = _load_saved_config(model_dir)
        stub = os.path.basename(model_dir)

        if stub in val_cache:
            entry = val_cache[stub]
        else:
            entry = _get_val_data(config, val_cache, stub)
        val_data = entry["data"]

        dv_config = config.training_config
        model = DVNet(
            dv_config["input_dim"], dv_config["hidden_dims"], dv_config["activation"]
        )
        state_path = os.path.join(full_dir, "dv_model.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing DV model at {state_path}")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        denom_mean = np.asarray(
            config.data_args["denominator_mean"], dtype=np.float32
        )
        denom_cov = np.asarray(
            config.data_args["denominator_cov"], dtype=np.float32
        )
        rng = np.random.RandomState(config.data_seed + 9876)

        errors = []
        with torch.no_grad():
            for _ in range(n_eval_trials):
                n_eval_p1 = min(eval_size, val_data.shape[0])
                idx = np.random.choice(val_data.shape[0], size=n_eval_p1, replace=False)
                batch_p1 = torch.from_numpy(val_data[idx]).float()

                n_eval_p0 = max(eval_size, 1024)
                neg_samples = rng.multivariate_normal(
                    denom_mean, denom_cov, size=n_eval_p0
                )
                batch_p0 = torch.from_numpy(neg_samples.astype(np.float32)).float()

                f_p1 = model(batch_p1).cpu().numpy()
                f_p0 = model(batch_p0).cpu().numpy()
                log_term = logsumexp(f_p0) - np.log(len(f_p0))
                est_kl = float(np.mean(f_p1) - log_term)
                rel_error = abs(est_kl - config.true_kl) / abs(config.true_kl)
                errors.append(rel_error)

        avg_error = float(np.mean(errors))
        grouped[(config.true_kl, int(config.data_args["n_samples"]))].append(avg_error)
        per_model.append(
            (model_dir, config.true_kl, int(config.data_args["n_samples"]), avg_error)
        )

    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))

    return per_model, summary, val_cache


def evaluate_nwj_models(model_dirs, eval_size, n_eval_trials, val_cache):
    grouped = defaultdict(list)
    per_model = []

    for model_dir in sorted(model_dirs):
        full_dir = os.path.join(project_root, "saved_models", model_dir)
        config = _load_saved_config(model_dir)
        stub = os.path.basename(model_dir)

        if stub in val_cache:
            entry = val_cache[stub]
        else:
            entry = _get_val_data(config, val_cache, stub)
        val_data = entry["data"]

        nwj_config = config.training_config

        # Choose the same critic architecture that was used during training.
        # Default: generic MLP critic.
        # model = NWJNet(
        #     nwj_config["input_dim"],
        #     nwj_config["hidden_dims"],
        #     nwj_config["activation"],
        # )
        # Alternatively, if you trained with the quadratic critic, comment the line
        # above and uncomment the line below.
        model = QuadraticNWJNet(nwj_config["input_dim"])

        state_path = os.path.join(full_dir, "nwj_model.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing NWJ model at {state_path}")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        denom_mean = np.asarray(
            config.data_args["denominator_mean"], dtype=np.float32
        )
        denom_cov = np.asarray(
            config.data_args["denominator_cov"], dtype=np.float32
        )
        rng = np.random.RandomState(config.data_seed + 2468)

        # clamp_min, clamp_max = -5.0, 5.0
        errors = []
        with torch.no_grad():
            for _ in range(n_eval_trials):
                # P (numerator) evaluation batch
                n_eval_p1 = min(eval_size, val_data.shape[0])
                idx = np.random.choice(val_data.shape[0], size=n_eval_p1, replace=False)
                batch_p1 = torch.from_numpy(val_data[idx]).float()

                # Q (denominator) evaluation batch
                n_eval_p0 = max(eval_size, 1024)
                neg_samples = rng.multivariate_normal(
                    denom_mean, denom_cov, size=n_eval_p0
                )
                batch_p0 = torch.from_numpy(neg_samples.astype(np.float32)).float()

                f_p1 = model(batch_p1)
                f_p0 = model(batch_p0)

                t_p1 = f_p1
                t_p0 = f_p0

                # t_p1 = torch.clamp(f_p1, clamp_min, clamp_max)
                # t_p0 = torch.clamp(f_p0, clamp_min, clamp_max)

                g_p1 = torch.exp(t_p1)
                g_p0 = torch.exp(t_p0)

                log_g_p1 = t_p1  # log g = t

                est_kl = float(log_g_p1.mean().item() - g_p0.mean().item() + 1.0)
                rel_error = abs(est_kl - config.true_kl) / abs(config.true_kl)
                errors.append(rel_error)

        avg_error = float(np.mean(errors))
        grouped[(config.true_kl, int(config.data_args["n_samples"]))].append(avg_error)
        per_model.append(
            (model_dir, config.true_kl, int(config.data_args["n_samples"]), avg_error)
        )

    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))

    return per_model, summary, val_cache


def evaluate_mdre_models(model_dirs, eval_size, n_eval_trials, val_cache):
    grouped = defaultdict(list)
    per_model = []

    for model_dir in sorted(model_dirs):
        full_dir = os.path.join(project_root, "saved_models", model_dir)
        config = _load_saved_config(model_dir)
        stub = os.path.basename(model_dir)

        if stub in val_cache:
            entry = val_cache[stub]
        else:
            entry = _get_val_data(config, val_cache, stub)
        val_data = entry["data"]

        mdre_config = config.training_config
        num_classes = mdre_config["num_classes"]
        model = MDRENet(
            input_dim=mdre_config["input_dim"],
            hidden_dims=mdre_config["hidden_dims"],
            activation=mdre_config["activation"],
            num_classes=num_classes,
        )
        state_path = os.path.join(full_dir, "mdre_model.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing MDRE model at {state_path}")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        errors = []
        with torch.no_grad():
            for _ in range(n_eval_trials):
                n_eval = min(eval_size, val_data.shape[0])
                idx = np.random.choice(val_data.shape[0], size=n_eval, replace=False)
                batch = torch.from_numpy(val_data[idx]).float()
                logits = model(batch)
                log_ratio = logits[:, 0] - logits[:, 1]
                est_kl = float(log_ratio.mean().item())
                rel_error = abs(est_kl - config.true_kl) / abs(config.true_kl)
                errors.append(rel_error)

        avg_error = float(np.mean(errors))
        grouped[(config.true_kl, int(config.data_args["n_samples"]))].append(avg_error)
        per_model.append((model_dir, config.true_kl, int(config.data_args["n_samples"]), avg_error))

    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))

    return per_model, summary, val_cache


def plot_results(result_dict, kl_targets, sample_sizes,
                 save_path=None, zoom_save_path=None,
                 save_path_pdf=None, zoom_save_path_pdf=None):
    """
    result_dict: {method: {eval_size: {true_kl: {n_samples: rel_err}}}}
    """

    def _plot(subset, title_suffix, png_path, pdf_path):
        if not subset:
            return
        plt.figure(figsize=(9, 4))
        method_colors = {
            "tdre": "#2ca02c",
            "bdre": "#1f77b4",
            "dv": "#ff7f0e",
            "nwj": "#d62728",
            "mdre": "#9467bd",
        }
        linestyles = {"tdre": "--", "bdre": "-"}
        eval_sizes_sorted = sorted(
            {e for method in result_dict.values() for e in method.keys()}
        )
        marker_cycle = ["o", "s", "^", "D", "v"]
        marker_map = {
            eval_size: marker_cycle[i % len(marker_cycle)]
            for i, eval_size in enumerate(eval_sizes_sorted)
        }

        for method, summaries in result_dict.items():
            for eval_size, summary in summaries.items():
                for kl in kl_targets:
                    if kl not in summary:
                        continue
                    ys = [summary.get(kl, {}).get(n, np.nan) for n in subset]
                    label = f"{method.upper()} (M={eval_size})"
                    plt.plot(
                        subset,
                        ys,
                        marker=marker_map.get(eval_size, "o"),
                        linestyle=linestyles.get(method, "-"),
                        color=method_colors.get(method, "#333333"),
                        label=label,
                    )

        plt.xlabel("Training sample size")
        plt.ylabel("Relative KL error (avg)")
        plt.title(f"TDRE / BDRE / DV / NWJ on 10D Gaussians{title_suffix}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if png_path:
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            plt.savefig(png_path, bbox_inches="tight", dpi=600)
            print(f"Saved plot to {png_path}")
        if pdf_path:
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved PDF plot to {pdf_path}")
        plt.show()

    # Global plot over all methods, all eval_sizes
    _plot(sample_sizes, "", save_path, save_path_pdf)

    # Global zoomed plot (n >= 1000)
    zoom_samples = [n for n in sample_sizes if n >= 1000]
    if zoom_samples:
        _plot(zoom_samples, " (Zoom: n ≥ 1000)", zoom_save_path, zoom_save_path_pdf)

    # ------------------------------------------------------------------
    # Additional per-method plots: for each method, show M lines
    # (M = eval_size, e.g. 10 / 100 / 1000) on the same axes.
    # ------------------------------------------------------------------
    def _method_path(base_path, method_key, suffix=""):
        if base_path is None:
            return None
        root, ext = os.path.splitext(base_path)
        return f"{root}_{method_key}{suffix}{ext}"

    for method, summaries in result_dict.items():
        if not summaries:
            continue

        plt.figure(figsize=(9, 4))
        eval_sizes_sorted = sorted(summaries.keys())
        marker_cycle = ["o", "s", "^", "D", "v", "x"]

        for i, eval_size in enumerate(eval_sizes_sorted):
            for kl in kl_targets:
                if kl not in summaries[eval_size]:
                    continue
                ys = [
                    summaries[eval_size].get(kl, {}).get(n, np.nan)
                    for n in sample_sizes
                ]
                label = f"M={eval_size}"
                plt.plot(
                    sample_sizes,
                    ys,
                    marker=marker_cycle[i % len(marker_cycle)],
                    linestyle="-",
                    label=label,
                )

        plt.xlabel("Training sample size")
        plt.ylabel("Relative KL error (avg)")
        plt.title(f"{method.upper()} (per eval_size) on 10D Gaussians")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        per_method_path = _method_path(save_path, method)
        if per_method_path:
            os.makedirs(os.path.dirname(per_method_path), exist_ok=True)
            plt.savefig(per_method_path, bbox_inches="tight")
            print(f"Saved per-method plot to {per_method_path}")
        plt.show()


def save_summary_text(result_dict, kl_targets, sample_sizes, eval_sizes, args, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("TDRE / BDRE / DV / NWJ / MDRE Evaluation Summary")
    lines.append(f"TDRE models: {args.tdre_model_base or args.tdre_model_dirs}")
    lines.append(f"BDRE models: {args.bdre_model_base or args.bdre_model_dirs}")
    lines.append(f"DV models: {args.dv_model_base or args.dv_model_dirs}")
    lines.append(f"NWJ models: {args.nwj_model_base or args.nwj_model_dirs}")
    lines.append(f"MDRE models: {args.mdre_model_base or args.mdre_model_dirs}")
    lines.append(f"KL targets: {kl_targets}")
    lines.append(f"Training sample sizes: {sample_sizes}")
    lines.append(f"Eval sizes: {eval_sizes}")
    lines.append(f"n_eval_trials: {args.n_eval_trials}")
    lines.append("")
    for method, summaries in result_dict.items():
        lines.append(f"Method: {method.upper()}")
        for eval_size in sorted(summaries.keys()):
            lines.append(f"  Eval size {eval_size}:")
            for kl in sorted(summaries[eval_size].keys()):
                entries = ", ".join(
                    f"n={n}: {summaries[eval_size][kl][n]:.4f}"
                    for n in sorted(summaries[eval_size][kl].keys())
                )
                lines.append(f"    KL {kl:.2f} -> {entries}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TDRE/BDRE/DV/NWJ/MDRE models on 10D Gaussian experiments."
    )
    parser.add_argument(
        "--tdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit TDRE dirs (relative to saved_models/)",
    )
    parser.add_argument(
        "--tdre_model_base",
        type=str,
        default=None,
        help="TDRE dataset/time_id prefix e.g. gaussians_10d/20250101-1200",
    )
    parser.add_argument(
        "--bdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit BDRE dirs (relative to saved_models/)",
    )
    parser.add_argument(
        "--bdre_model_base",
        type=str,
        default=None,
        help="BDRE dataset/time_id prefix e.g. bdre_gaussians_10d/20250101-1200",
    )
    parser.add_argument(
        "--dv_model_dirs",
        nargs="+",
        default=None,
        help="Explicit DV dirs (relative to saved_models/)",
    )
    parser.add_argument(
        "--dv_model_base",
        type=str,
        default=None,
        help="DV dataset/time_id prefix e.g. dv_gaussians_10d/20250101-1200",
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=None,
        help="Backward-compatible alias for --tdre_model_dirs",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Backward-compatible alias for --tdre_model_base",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=[10, 100, 1000],
        help="[Deprecated] Single evaluation size. Prefer --eval_sizes.",
    )
    parser.add_argument(
        "--eval_sizes",
        type=int,
        nargs="+",
        default=[10, 100, 1000],
        help="One or more evaluation batch sizes to compare.",
    )
    parser.add_argument(
        "--n_eval_trials",
        type=int,
        default=20,
        help="Number of Monte Carlo evaluations per model.",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[10, 100, 250, 500, 1000, 1500, 2000, 3000],
        help="Training sample sizes to plot on x-axis.",
    )
    parser.add_argument(
        "--kl_targets",
        type=float,
        nargs="+",
        default=[20.0],
        help="True KL targets to plot.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="results/gauss10d/tdre_bdre_dv_nwj.png",
        help="Path (relative to repo root) to save the primary plot.",
    )
    parser.add_argument(
        "--save_plot_zoom",
        type=str,
        default="results/gauss10d/tdre_bdre_dv_nwj_zoom.png",
        help="Where to save the zoomed plot (n>=1000). Leave empty to skip.",
    )
    parser.add_argument(
        "--save_plot_pdf",
        type=str,
        default="results/gauss10d/tdre_bdre_dv_nwj.pdf",
        help="Path to save the primary plot as PDF.",
    )
    parser.add_argument(
        "--save_plot_zoom_pdf",
        type=str,
        default="results/gauss10d/tdre_bdre_dv_nwj_zoom.pdf",
        help="Path to save the zoomed plot as PDF.",
    )
    parser.add_argument(
        "--save_summary",
        type=str,
        default="results/gauss10d/eval_summary.txt",
        help="Path for the textual summary (relative to repo root).",
    )
    parser.add_argument(
        "--nwj_model_dirs",
        nargs="+",
        default=None,
        help="Explicit NWJ dirs (relative to saved_models/)",
    )
    parser.add_argument(
        "--nwj_model_base",
        type=str,
        default=None,
        help="NWJ dataset/time_id prefix e.g. nwj_gaussians_10d/20250101-1200",
    )
    parser.add_argument(
        "--mdre_model_dirs",
        nargs="+",
        default=None,
        help="Explicit MDRE dirs (relative to saved_models/)",
    )
    parser.add_argument(
        "--mdre_model_base",
        type=str,
        default=None,
        help="MDRE dataset/time_id prefix e.g. mdre_gaussians_10d/20250101-1200",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tdre_dirs = _collect_model_dirs(
        args.tdre_model_dirs or args.model_dirs,
        args.tdre_model_base or args.model_base,
    )
    bdre_dirs = _collect_model_dirs(args.bdre_model_dirs, args.bdre_model_base)
    dv_dirs = _collect_model_dirs(args.dv_model_dirs, args.dv_model_base)
    nwj_dirs = _collect_model_dirs(args.nwj_model_dirs, args.nwj_model_base)
    mdre_dirs = _collect_model_dirs(args.mdre_model_dirs, args.mdre_model_base)

    if (
        not tdre_dirs
        and not bdre_dirs
        and not dv_dirs
        and not nwj_dirs
        and not mdre_dirs
    ):
        raise ValueError(
            "Provide TDRE, BDRE, DV, NWJ, and/or MDRE model directories to evaluate."
        )

    eval_sizes = args.eval_sizes if args.eval_sizes else [args.eval_size]
    val_cache = {}
    results = {}

    for eval_size in eval_sizes:
        print(f"\n=== Evaluating with eval_size={eval_size} ===")
        if tdre_dirs:
            per_model, summary, val_cache = evaluate_tdre_models(
                tdre_dirs, eval_size, args.n_eval_trials, val_cache
            )
            results.setdefault("tdre", {})[eval_size] = summary
            print("TDRE per-model results (dir, true_KL, n_samples, rel_err):")
            for model_dir, kl, n_samples, err in per_model:
                print(
                    f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}"
                )
        if bdre_dirs:
            per_model, summary, val_cache = evaluate_bdre_models(
                bdre_dirs, eval_size, args.n_eval_trials, val_cache
            )
            results.setdefault("bdre", {})[eval_size] = summary
            print("BDRE per-model results (dir, true_KL, n_samples, rel_err):")
            for model_dir, kl, n_samples, err in per_model:
                print(
                    f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}"
                )
        if dv_dirs:
            per_model, summary, val_cache = evaluate_dv_models(
                dv_dirs, eval_size, args.n_eval_trials, val_cache
            )
            results.setdefault("dv", {})[eval_size] = summary
            print("DV per-model results (dir, true_KL, n_samples, rel_err):")
            for model_dir, kl, n_samples, err in per_model:
                print(
                    f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}"
                )
        if nwj_dirs:
            per_model, summary, val_cache = evaluate_nwj_models(
                nwj_dirs, eval_size, args.n_eval_trials, val_cache
            )
            results.setdefault("nwj", {})[eval_size] = summary
            print("NWJ per-model results (dir, true_KL, n_samples, rel_err):")
            for model_dir, kl, n_samples, err in per_model:
                print(
                    f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}"
                )
        if mdre_dirs:
            per_model, summary, val_cache = evaluate_mdre_models(
                mdre_dirs, eval_size, args.n_eval_trials, val_cache
            )
            results.setdefault("mdre", {})[eval_size] = summary
            print("MDRE per-model results (dir, true_KL, n_samples, rel_err):")
            for model_dir, kl, n_samples, err in per_model:
                print(
                    f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}"
                )

        for method, summary in results.items():
            if eval_size not in summary:
                continue
            print(f"\nAveraged {method.upper()} results for eval_size={eval_size}:")
            for kl in sorted(summary[eval_size].keys()):
                entries = ", ".join(
                    f"n={n}: {summary[eval_size][kl][n]:.4f}"
                    for n in sorted(summary[eval_size][kl].keys())
                )
                print(f"  KL {kl:.2f} -> {entries}")

    plot_path = (
        os.path.join(project_root, args.save_plot) if args.save_plot else None
    )
    zoom_path = (
        os.path.join(project_root, args.save_plot_zoom)
        if args.save_plot_zoom
        else None
    )
    plot_pdf_path = (
        os.path.join(project_root, args.save_plot_pdf)
        if args.save_plot_pdf
        else None
    )
    zoom_pdf_path = (
        os.path.join(project_root, args.save_plot_zoom_pdf)
        if args.save_plot_zoom_pdf
        else None
    )
    plot_results(
        results,
        args.kl_targets,
        args.sample_sizes,
        save_path=plot_path,
        zoom_save_path=zoom_path,
        save_path_pdf=plot_pdf_path,
        zoom_save_path_pdf=zoom_pdf_path,
    )
    if args.save_summary:
        summary_path = os.path.join(project_root, args.save_summary)
        eval_sizes_used = args.eval_sizes if args.eval_sizes else [args.eval_size]
        save_summary_text(
            results,
            args.kl_targets,
            args.sample_sizes,
            eval_sizes_used,
            args,
            summary_path,
        )


if __name__ == "__main__":
    main()
