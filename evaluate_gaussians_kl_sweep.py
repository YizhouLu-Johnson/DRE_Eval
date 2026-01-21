import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_gaussians_10d import (
    _collect_model_dirs,
    evaluate_bdre_models,
    evaluate_dv_models,
    evaluate_mdre_models,
    evaluate_nwj_models,
    evaluate_tdre_models,
)
from utils.experiment_utils import project_root
from utils.misc_utils import AttrDict

# TSM imports (PyTorch-based)
try:
    import torch
    from train_tsm_10d_tailored import (
        TimeScoreNetwork,
        TSMGaussianDataset,
        compute_density_ratios,
    )
    TSM_AVAILABLE = True
except ImportError:
    TSM_AVAILABLE = False
    print("Warning: TSM evaluation not available. Install PyTorch to enable.")


def _load_saved_config_tsm(model_dir):
    """Load TSM model config."""
    cfg_path = os.path.join(project_root, "saved_models", model_dir, "config.json")
    with open(cfg_path, "r") as f:
        return AttrDict(json.load(f))


def _true_kl_tsm(config):
    """Extract true KL from TSM config."""
    data_args = config.data_args
    return float(
        data_args.get(
            "analytic_kl",
            data_args.get(
                "target_kl",
                data_args.get(
                    "true_mutual_info",
                    data_args.get("true_kl", config.get("true_kl"))
                ),
            ),
        )
    )


def evaluate_tsm_models(model_dirs, eval_size, n_eval_trials, val_cache):
    """
    Evaluate TSM (Time Score Matching) models.
    
    This mirrors the interface of other evaluate_*_models functions.
    """
    if not TSM_AVAILABLE:
        print("TSM evaluation skipped: PyTorch not available")
        return [], {}, val_cache
    
    grouped = defaultdict(list)
    per_model = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_dir in sorted(model_dirs):
        stub = os.path.basename(model_dir)
        config = _load_saved_config_tsm(model_dir)
        
        # Get data config
        data_args = config.data_args
        n_dims = int(data_args.get("n_dims", 10))
        seed = int(config.get("data_seed", 0))
        
        # Create dataset for evaluation samples
        dataset = TSMGaussianDataset(
            data_args=data_args,
            n_dims=n_dims,
            seed=seed,
            device=device
        )
        
        # Load model
        model_path = os.path.join(
            project_root, "saved_models", model_dir, "tsm_model_best.pt"
        )
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping")
            continue
        
        training_config = config.training_config
        model = TimeScoreNetwork(
            input_dim=training_config["input_dim"],
            hidden_dim=training_config["hidden_dim"],
            n_layers=training_config["n_layers"]
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        errors = []
        true_kl = _true_kl_tsm(config)
        
        for _ in range(n_eval_trials):
            # Sample from validation/test set
            val_data = dataset.val_q_samples
            n_eval = min(eval_size, len(val_data))
            idx = np.random.choice(len(val_data), size=n_eval, replace=False)
            batch = val_data[idx].numpy()
            
            # Compute density ratios (negate: TSM convention gives log(p/q))
            log_ratios, _ = compute_density_ratios(model, batch)
            est_kl = float(-np.mean(log_ratios))
            rel_error = abs(est_kl - true_kl) / true_kl
            errors.append(rel_error)
        
        avg_error = float(np.mean(errors))
        n_samples = int(data_args.get("n_samples", data_args.get("train_sample_size", 3000)))
        grouped[(true_kl, n_samples)].append(avg_error)
        per_model.append((model_dir, true_kl, n_samples, avg_error))
    
    # Build summary
    summary = {}
    for (true_kl, n_samples), errs in grouped.items():
        summary.setdefault(true_kl, {})[n_samples] = float(np.mean(errs))
    
    return per_model, summary, val_cache


EVAL_FUNCS = {
    "tdre": evaluate_tdre_models,
    "bdre": evaluate_bdre_models,
    "dv": evaluate_dv_models,
    "nwj": evaluate_nwj_models,
    "mdre": evaluate_mdre_models,
    "tsm": evaluate_tsm_models,
}

DEFAULT_PATTERNS = {
    "tdre": "tre_gaussians_10d_kl{kl}/kl{kl}",
    "bdre": "bdre_gaussians_10d_kl{kl}/kl{kl}",
    "dv": "dv_gaussians_10d_kl{kl}/kl{kl}",
    "nwj": "nwj_gaussians_10d_kl{kl}/kl{kl}",
    "mdre": "mdre_gaussians_10d_kl{kl}/kl{kl}",
    "tsm": "tsm_gaussians_10d_kl{kl}/kl{kl}",
}


def plot_vs_kl(errors_by_method, target_kls, save_path=None, save_path_pdf=None):
    if not errors_by_method:
        return
    plt.figure(figsize=(8, 4))
    markers = {
        "tdre": "o",
        "bdre": "o",
        "dv": "o",
        "nwj": "o",
        "mdre": "o",
        "tsm": "s",  # square for TSM
    }
    colors = {
        "tdre": "#2ca02c",
        "bdre": "#1f77b4",
        "dv": "#ff7f0e",
        "nwj": "#d62728",
        "mdre": "#9467bd",
        "tsm": "#17becf",  # cyan for TSM
    }
    for method, err_map in errors_by_method.items():
        ys = [err_map.get(kl, float("nan")) for kl in target_kls]
        plt.plot(
            target_kls,
            ys,
            marker=markers.get(method, "o"),
            color=colors.get(method, "#555555"),
            markersize=4,
            label=method.upper(),
        )

    plt.xlabel("True KL divergence (nats)")
    plt.ylabel("Relative KL error (avg)")
    plt.title("Relative error vs. true KL (M=1000)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"Saved KL-sweep plot to {save_path}")
    if save_path_pdf:
        os.makedirs(os.path.dirname(save_path_pdf), exist_ok=True)
        plt.savefig(save_path_pdf, bbox_inches="tight")
        print(f"Saved KL-sweep PDF plot to {save_path_pdf}")
    plt.show()


def save_summary(errors_by_method, target_kls, args, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("KL Sweep Summary (fixed eval_size)")
    lines.append(f"eval_size: {args.eval_size}")
    lines.append(f"sample_size_focus: {args.sample_size_focus}")
    lines.append(f"target_kls: {target_kls}")
    lines.append(f"tdre: {args.tdre_model_base or args.tdre_model_dirs}")
    lines.append(f"bdre: {args.bdre_model_base or args.bdre_model_dirs}")
    lines.append(f"dv: {args.dv_model_base or args.dv_model_dirs}")
    lines.append(f"nwj: {args.nwj_model_base or args.nwj_model_dirs}")
    lines.append(f"mdre: {args.mdre_model_base or args.mdre_model_dirs}")
    lines.append(f"tsm: {getattr(args, 'tsm_model_base', None) or getattr(args, 'tsm_model_dirs', None)}")
    lines.append("")
    for method, err_map in errors_by_method.items():
        lines.append(method.upper())
        for kl in target_kls:
            val = err_map.get(kl)
            if val is None:
                lines.append(f"  KL {kl:.2f}: unavailable")
            else:
                lines.append(f"  KL {kl:.2f}: {val:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved KL sweep summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot relative KL error vs. true KL for fixed sample size."
    )
    parser.add_argument("--target_kls", type=float, nargs="+", default=[10,20,30,40,50,60,70,80])
    parser.add_argument(
        "--sample_size_focus",
        type=int,
        default=3000,
        help="Training sample size to extract from the summaries.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Evaluation batch size (M).",
    )
    parser.add_argument(
        "--n_eval_trials",
        type=int,
        default=20,
        help="Number of Monte Carlo evaluation batches per model.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="results/kl_sweep/kl_sweep.png",
        help="PNG path relative to repo root.",
    )
    parser.add_argument(
        "--save_plot_pdf",
        type=str,
        default="results/kl_sweep/kl_sweep.pdf",
        help="PDF path relative to repo root.",
    )
    parser.add_argument(
        "--save_summary",
        type=str,
        default="results/kl_sweep/kl_sweep_summary.txt",
        help="Text summary path relative to repo root.",
    )
    # model location flags (mirror evaluate_gaussians_10d.py)
    parser.add_argument("--tdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--tdre_model_base", type=str, default=None)
    parser.add_argument("--bdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--bdre_model_base", type=str, default=None)
    parser.add_argument("--dv_model_dirs", nargs="+", default=None)
    parser.add_argument("--dv_model_base", type=str, default=None)
    parser.add_argument("--nwj_model_dirs", nargs="+", default=None)
    parser.add_argument("--nwj_model_base", type=str, default=None)
    parser.add_argument("--mdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--mdre_model_base", type=str, default=None)
    parser.add_argument("--tsm_model_dirs", nargs="+", default=None)
    parser.add_argument("--tsm_model_base", type=str, default=None)
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=None,
        help="Backward-compatible alias for --tdre_model_dirs.",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Backward-compatible alias for --tdre_model_base.",
    )
    return parser.parse_args()


def _match_summary_entry(summary, target_kl, sample_size, tol=1e-6):
    """
    Fetch summary value while being tolerant to floating-point KL keys.
    """
    # Exact match first
    if target_kl in summary and sample_size in summary[target_kl]:
        return summary[target_kl][sample_size]

    # Otherwise search for a KL key within tolerance
    for kl_key, sample_dict in summary.items():
        if abs(kl_key - target_kl) <= tol and sample_size in sample_dict:
            return sample_dict[sample_size]
    return None


def _collect_dirs_for_method(
    explicit_dirs,
    base_prefix,
    target_kls,
    default_pattern,
):
    """
    Determine which saved-model directories to use for a method.
    Priority:
        1. If explicit directories specified, return them.
        2. Else if a single base prefix provided (dataset/time_id), use that.
        3. Otherwise, auto-expand the default pattern for each KL in target_kls.
    """
    if explicit_dirs:
        return explicit_dirs

    dirs = []
    if base_prefix:
        dirs = _collect_model_dirs(None, base_prefix)
    else:
        for kl in target_kls:
            base = default_pattern.format(kl=int(kl))
            dirs.extend(_collect_model_dirs(None, base))
    return dirs


def main():
    args = parse_args()
    dir_map = {
        "tdre": _collect_dirs_for_method(
            args.tdre_model_dirs or args.model_dirs,
            args.tdre_model_base or args.model_base,
            args.target_kls,
            DEFAULT_PATTERNS["tdre"],
        ),
        "bdre": _collect_dirs_for_method(
            args.bdre_model_dirs,
            args.bdre_model_base,
            args.target_kls,
            DEFAULT_PATTERNS["bdre"],
        ),
        "dv": _collect_dirs_for_method(
            args.dv_model_dirs,
            args.dv_model_base,
            args.target_kls,
            DEFAULT_PATTERNS["dv"],
        ),
        "nwj": _collect_dirs_for_method(
            args.nwj_model_dirs,
            args.nwj_model_base,
            args.target_kls,
            DEFAULT_PATTERNS["nwj"],
        ),
        "mdre": _collect_dirs_for_method(
            args.mdre_model_dirs,
            args.mdre_model_base,
            args.target_kls,
            DEFAULT_PATTERNS["mdre"],
        ),
        "tsm": _collect_dirs_for_method(
            args.tsm_model_dirs,
            args.tsm_model_base,
            args.target_kls,
            DEFAULT_PATTERNS["tsm"],
        ),
    }
    if not any(dir_map.values()):
        raise ValueError("Provide at least one set of model directories.")

    val_cache = {}
    errors_by_method = defaultdict(dict)
    eval_size = args.eval_size

    for method, model_dirs in dir_map.items():
        if not model_dirs:
            continue
        eval_fn = EVAL_FUNCS[method]
        per_model, summary, val_cache = eval_fn(
            model_dirs, eval_size, args.n_eval_trials, val_cache
        )
        print(f"\n{method.upper()} per-model results (eval_size={eval_size}):")
        for model_dir, kl, n_samples, err in per_model:
            print(f"  {model_dir}: KL={kl:.2f}, n={n_samples}, rel_err={err:.4f}")
        for kl in args.target_kls:
            err = _match_summary_entry(summary, kl, args.sample_size_focus)
            if err is None:
                print(
                    f"Warning: missing data for method {method} "
                    f"(KL={kl}, n={args.sample_size_focus})."
                )
                continue
            errors_by_method[method][kl] = err

    save_path = os.path.join(project_root, args.save_plot) if args.save_plot else None
    save_path_pdf = (
        os.path.join(project_root, args.save_plot_pdf) if args.save_plot_pdf else None
    )
    plot_vs_kl(errors_by_method, args.target_kls, save_path, save_path_pdf)

    if args.save_summary:
        summary_path = os.path.join(project_root, args.save_summary)
        save_summary(errors_by_method, args.target_kls, args, summary_path)


if __name__ == "__main__":
    main()
