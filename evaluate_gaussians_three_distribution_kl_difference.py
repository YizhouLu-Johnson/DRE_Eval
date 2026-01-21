import argparse
import os
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from matplotlib import colors as mpl_colors

tf.disable_v2_behavior()

from evaluate_gaussians_10d import (
    _collect_model_dirs,
    _get_val_data,
    _load_saved_config,
)
from evaluate_gaussians_three_distribution import (
    collect_method_dirs,
    tdre_estimates,
    bdre_estimates,
    dv_estimates,
    nwj_estimates,
    mdre_estimates,
    tsm_estimates,
    format_output_path,
)
from utils.experiment_utils import project_root
from utils.distribution_utils import (
    DIRICHLET,
    GAUSSIAN,
    get_distribution_params,
    infer_distribution_family,
    kl_divergence,
    sample_distribution,
)


KL_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]


def _extract_params(config, role):
    family, params = get_distribution_params(config.data_args, role)
    return family, params


def _infer_shared_family(local_cfg, distant_cfg):
    fam_local = infer_distribution_family(local_cfg.data_args)
    fam_distant = infer_distribution_family(distant_cfg.data_args)
    if fam_local != fam_distant:
        raise ValueError(
            f"Distribution mismatch between configs ({fam_local} vs {fam_distant})."
        )
    return fam_local


def _default_roots(method, is_local):
    suffix = "p1" if is_local else "p0"
    return [f"{method}_pstar_{suffix}_kl{kl}" for kl in KL_VALUES]


def _stub(path):
    return os.path.basename(path.rstrip("/"))


def _build_stub_map(dirs):
    return {_stub(d): d for d in dirs}


def _expand_root_dirs(root_list):
    if not root_list:
        return []
    results = []
    base_dir = Path(project_root) / "saved_models"
    for root in root_list:
        root_path = base_dir / root
        if not root_path.exists():
            print(f"Warning: root directory {root_path} not found.")
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir():
                cfg_path = child / "config.json"
                if cfg_path.exists():
                    results.append(f"{root}/{child.name}")
    return results


def _merge_dirs(primary, secondary):
    merged = []
    for seq in (primary, secondary):
        for item in seq:
            if item not in merged:
                merged.append(item)
    return merged


def _ensure_tdre_checkpoint_dir(config):
    """Ensure config.save_dir contains a usable 'model' subdirectory."""
    save_dir = config.save_dir
    model_dir = os.path.join(save_dir, "model")
    if os.path.isdir(model_dir):
        return
    if os.path.islink(model_dir):
        os.unlink(model_dir)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"TDRE save_dir does not exist: {save_dir}")
    candidates = []
    for child in os.listdir(save_dir):
        full = os.path.join(save_dir, child)
        if child.lower().endswith("model") and os.path.isdir(full):
            candidates.append(full)
    if not candidates:
        parent = os.path.dirname(save_dir)
        sibling = os.path.join(parent, os.path.basename(save_dir) + "model")
        if os.path.isdir(sibling):
            candidates.append(sibling)
        else:
            raise FileNotFoundError(
                f"No checkpoint directory found inside {save_dir}; "
                f"expected 'model/' or '*model/'. Checked sibling {sibling}."
            )
    target = sorted(candidates)[0]
    os.symlink(target, model_dir)


def estimate_method(method, local_info, distant_info, samples):
    def _tdre(info, smp):
        return tdre_estimates(
            info["path"],
            info["config"],
            info["val_entry"]["dp"],
            smp,
        )

    estimator_lookup = {
        "tdre": _tdre,
        "bdre": lambda info, smp: bdre_estimates(info["path"], info["config"], smp),
        "dv": lambda info, smp: dv_estimates(info["path"], info["config"], smp),
        "nwj": lambda info, smp: nwj_estimates(info["path"], info["config"], smp),
        "mdre": lambda info, smp: mdre_estimates(info["path"], info["config"], smp),
        "tsm": lambda info, smp: tsm_estimates(info["path"], info["config"], smp),
    }

    if method not in estimator_lookup:
        raise ValueError(f"Unsupported method {method}")

    local_vals = estimator_lookup[method](local_info, samples)
    distant_vals = estimator_lookup[method](distant_info, samples)
    local_mean = float(np.mean(local_vals))
    distant_mean = float(np.mean(distant_vals))
    return local_mean, distant_mean, local_mean - distant_mean, list(local_vals), list(distant_vals)


def plot_method(records, method, kl_shift, png_path=None, pdf_path=None, shift_tol=1e-6):
    subset = [r for r in records if abs(r["kl_shift"] - kl_shift) < shift_tol]
    if not subset:
        return
    xs = [r["kl_p0_p1"] for r in subset]
    ys = [r["kl_p0_pstar"] for r in subset]
    errs = [r["rel_error"] for r in subset]

    plt.figure(figsize=(5.2, 3.8))
    cmap = plt.get_cmap("coolwarm")
    norm = mpl_colors.Normalize(vmin=min(errs), vmax=max(errs))
    
    # Plot the scatter points (keep all points visible)
    sc = plt.scatter(xs, ys, c=errs, cmap=cmap, norm=norm, s=50, edgecolors="k", linewidths=0.4)
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1e-6, 0.04 * (x_max - x_min))
    y_pad = max(1e-6, 0.06 * (y_max - y_min))
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    # --- FIX START: Aggregate annotations by X-coordinate ---
    from collections import defaultdict
    grouped_data = defaultdict(list)
    
    # Group errors by their X position (rounded to avoid float mismatch)
    for x, y, err in zip(xs, ys, errs):
        x_key = round(x, 4) 
        grouped_data[x_key].append((y, err))

    # Annotate only once per X-position using the MEAN error
    for x_key, values in grouped_data.items():
        # Average Y position for the label anchor
        mean_y = np.mean([v[0] for v in values])
        # Average Error for the text
        mean_err = np.mean([v[1] for v in values])
        
        plt.annotate(
            f"{mean_err:.3f}",
            (x_key, mean_y),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            ha="center",
            va="bottom",
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.75, linewidth=0) # Added background to make text pop
        )
    # --- FIX END ---

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    plt.xlabel("KL(P0 || P1)", fontsize=10)
    plt.ylabel("KL(P0 || P*)", fontsize=10)
    plt.title(f"{method.upper()} (KL(P1||P*)={kl_shift})", fontsize=11)
    plt.grid(alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("Relative error of R", fontsize=10)
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


def plot_estimated_trials(records, method, kl_shift, png_path=None, pdf_path=None, shift_tol=1e-6):
    subset = [r for r in records if abs(r["kl_shift"] - kl_shift) < shift_tol]
    xs, ys, errs = [], [], []
    for rec in subset:
        trials_p0 = rec.get("trial_estimates_p0", [])
        trial_errs = rec.get("trial_rel_errors", [])
        for est_p0, err in zip(trials_p0, trial_errs):
            xs.append(rec["kl_p0_p1"])
            ys.append(est_p0)
            errs.append(err)
    if not xs:
        return
    plt.figure(figsize=(5.2, 3.8))
    cmap = plt.get_cmap("coolwarm")
    norm = mpl_colors.Normalize(vmin=min(errs), vmax=max(errs))
    sc = plt.scatter(xs, ys, c=errs, cmap=cmap, norm=norm, s=32, edgecolors="k", linewidths=0.3)
    true_pairs = {}
    for rec in subset:
        true_pairs[rec["kl_p0_p1"]] = rec["kl_pstar_p0"]
    first_line = True
    for x_val in sorted(true_pairs.keys()):
        y_val = true_pairs[x_val]
        label = "True KL(P*||P0)" if first_line else None
        plt.hlines(
            y_val,
            0.0,
            x_val,
            colors="black",
            linestyles="dotted",
            linewidth=1.2,
            label=label,
        )
        first_line = False
    plt.xlabel("KL(P0 || P1)", fontsize=10)
    plt.ylabel("Estimated KL(P* || P0)", fontsize=10)
    plt.title(f"{method.upper()} Estimated KL(P*||P0) (shift={kl_shift})", fontsize=11)
    plt.grid(alpha=0.3)
    x_min, x_max = plt.xlim()
    plt.xlim(left=0.0, right=x_max)
    cbar = plt.colorbar(sc)
    cbar.set_label("Relative error of KL(P*||P0)", fontsize=10)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    if png_path:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, bbox_inches="tight", dpi=600)
        print(f"Saved estimated plot to {png_path}")
    if pdf_path:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved estimated plot to {pdf_path}")
    plt.show()


def save_summary(method, records, kl_shifts, path):
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"{method.upper()} KL-difference evaluation summary",
        f"KL shifts: {kl_shifts}",
    ]
    lines.append("")
    for rec in records:
        lines.append(f"Model stub: {rec['stub']}")
        lines.append(
            "  KL(P0||P1)={:.4f}, KL(P0||P*)={:.4f}, KL(P*||P1)={:.4f}, KL(P*||P0)={:.4f}".format(
                rec["kl_p0_p1"],
                rec["kl_p0_pstar"],
                rec["kl_pstar_p1"],
                rec["kl_pstar_p0"],
            )
        )
        lines.append(
            "  Est KL(P*||P1)={:.5f} (true {:.5f}), Est KL(P*||P0)={:.5f} (true {:.5f})".format(
                rec["est_kl_pstar_p1"],
                rec["kl_pstar_p1"],
                rec["est_kl_pstar_p0"],
                rec["kl_pstar_p0"],
            )
        )
        lines.append(
            "  True Δ={:.5f}, Est Δ={:.5f}, Abs error={:.5f}, Rel error={:.5f}".format(
                rec["true_r"], rec["est_r"], rec["abs_error"], rec["rel_error"]
            )
        )
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate three-distribution KL difference via KL(P*||P1) - KL(P*||P0)."
    )
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--n_eval_trials", type=int, default=20)
    parser.add_argument("--kl_shift", type=float, default=0.5)
    parser.add_argument("--kl_shifts", type=float, nargs="+", default=None)
    parser.add_argument("--seed_offset", type=int, default=98765)

    # TDRE
    parser.add_argument("--local_tdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_tdre_model_base", type=str, default=None)
    parser.add_argument("--distant_tdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_tdre_model_base", type=str, default=None)
    parser.add_argument("--local_tdre_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_tdre_model_roots", nargs="+", default=None)

    # BDRE
    parser.add_argument("--local_bdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_bdre_model_base", type=str, default=None)
    parser.add_argument("--distant_bdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_bdre_model_base", type=str, default=None)
    parser.add_argument("--local_bdre_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_bdre_model_roots", nargs="+", default=None)

    # DV
    parser.add_argument("--local_dv_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_dv_model_base", type=str, default=None)
    parser.add_argument("--distant_dv_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_dv_model_base", type=str, default=None)
    parser.add_argument("--local_dv_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_dv_model_roots", nargs="+", default=None)

    # NWJ
    parser.add_argument("--local_nwj_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_nwj_model_base", type=str, default=None)
    parser.add_argument("--distant_nwj_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_nwj_model_base", type=str, default=None)
    parser.add_argument("--local_nwj_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_nwj_model_roots", nargs="+", default=None)

    # MDRE
    parser.add_argument("--local_mdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_mdre_model_base", type=str, default=None)
    parser.add_argument("--distant_mdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_mdre_model_base", type=str, default=None)
    parser.add_argument("--local_mdre_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_mdre_model_roots", nargs="+", default=None)

    # TSM
    parser.add_argument("--local_tsm_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_tsm_model_base", type=str, default=None)
    parser.add_argument("--distant_tsm_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_tsm_model_base", type=str, default=None)
    parser.add_argument("--local_tsm_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_tsm_model_roots", nargs="+", default=None)

    parser.add_argument("--save_plot", type=str, default="results/three_dist_kldiff/plot.png")
    parser.add_argument("--save_plot_pdf", type=str, default="results/three_dist_kldiff/plot.pdf")
    parser.add_argument("--save_summary", type=str, default="results/three_dist_kldiff/summary.txt")
    return parser.parse_args()


def gather_method_pairs(
    local_dirs,
    local_base,
    local_roots,
    distant_dirs,
    distant_base,
    distant_roots,
):
    local = _collect_model_dirs(local_dirs, local_base)
    local = _merge_dirs(local, _expand_root_dirs(local_roots))
    distant = _collect_model_dirs(distant_dirs, distant_base)
    distant = _merge_dirs(distant, _expand_root_dirs(distant_roots))
    local_map = _build_stub_map(local)
    distant_map = _build_stub_map(distant)
    common = sorted(local_map.keys() & distant_map.keys())
    return {stub: (local_map[stub], distant_map[stub]) for stub in common}


def build_bundle_from_configs(
    stub,
    local_cfg,
    distant_cfg,
    eval_size,
    n_eval_trials,
    seed_offset,
    cache,
):
    family = _infer_shared_family(local_cfg, distant_cfg)
    _, params_star = _extract_params(local_cfg, "numerator")
    _, params_p1 = _extract_params(local_cfg, "denominator")
    _, params_p0 = _extract_params(distant_cfg, "denominator")

    kl_p0_p1 = kl_divergence(params_p0, params_p1, family)
    kl_p0_pstar = kl_divergence(params_p0, params_star, family)
    kl_pstar_p1 = kl_divergence(params_star, params_p1, family)
    kl_pstar_p0 = kl_divergence(params_star, params_p0, family)
    kl_shift = kl_divergence(params_p1, params_star, family)
    true_r = kl_pstar_p1 - kl_pstar_p0

    key = (stub, round(float(kl_shift), 6), eval_size, n_eval_trials)
    if key in cache:
        return cache[key]

    seed = int(local_cfg.data_seed + seed_offset + kl_shift * 1000.0)
    rng = np.random.RandomState(seed)
    samples = [
        sample_distribution(family, params_star, eval_size, rng)
        for _ in range(n_eval_trials)
    ]

    bundle = {
        "kl_p0_p1": kl_p0_p1,
        "kl_p0_pstar": kl_p0_pstar,
        "kl_pstar_p1": kl_pstar_p1,
        "kl_pstar_p0": kl_pstar_p0,
        "kl_shift": kl_shift,
        "true_r": true_r,
        "samples": samples,
    }
    cache[key] = bundle
    return bundle


def main():
    args = parse_args()
    requested_shifts = args.kl_shifts if args.kl_shifts else [args.kl_shift]
    requested_shifts = [float(s) for s in requested_shifts]
    shift_tol = 1e-3

    local_tdre_roots = args.local_tdre_model_roots or _default_roots("tdre", True)
    distant_tdre_roots = args.distant_tdre_model_roots or _default_roots("tdre", False)
    local_bdre_roots = args.local_bdre_model_roots or _default_roots("bdre", True)
    distant_bdre_roots = args.distant_bdre_model_roots or _default_roots("bdre", False)
    local_dv_roots = args.local_dv_model_roots or _default_roots("dv", True)
    distant_dv_roots = args.distant_dv_model_roots or _default_roots("dv", False)
    local_nwj_roots = args.local_nwj_model_roots or _default_roots("nwj", True)
    distant_nwj_roots = args.distant_nwj_model_roots or _default_roots("nwj", False)
    local_mdre_roots = args.local_mdre_model_roots or _default_roots("mdre", True)
    distant_mdre_roots = args.distant_mdre_model_roots or _default_roots("mdre", False)
    local_tsm_roots = args.local_tsm_model_roots or _default_roots("tsm", True)
    distant_tsm_roots = args.distant_tsm_model_roots or _default_roots("tsm", False)

    method_pairs = {
        "tdre": gather_method_pairs(
            args.local_tdre_model_dirs, args.local_tdre_model_base,
            local_tdre_roots,
            args.distant_tdre_model_dirs, args.distant_tdre_model_base,
            distant_tdre_roots),
        "bdre": gather_method_pairs(
            args.local_bdre_model_dirs, args.local_bdre_model_base,
            local_bdre_roots,
            args.distant_bdre_model_dirs, args.distant_bdre_model_base,
            distant_bdre_roots),
        "dv": gather_method_pairs(
            args.local_dv_model_dirs, args.local_dv_model_base,
            local_dv_roots,
            args.distant_dv_model_dirs, args.distant_dv_model_base,
            distant_dv_roots),
        "nwj": gather_method_pairs(
            args.local_nwj_model_dirs, args.local_nwj_model_base,
            local_nwj_roots,
            args.distant_nwj_model_dirs, args.distant_nwj_model_base,
            distant_nwj_roots),
        "mdre": gather_method_pairs(
            args.local_mdre_model_dirs, args.local_mdre_model_base,
            local_mdre_roots,
            args.distant_mdre_model_dirs, args.distant_mdre_model_base,
            distant_mdre_roots),
        "tsm": gather_method_pairs(
            args.local_tsm_model_dirs, args.local_tsm_model_base,
            local_tsm_roots,
            args.distant_tsm_model_dirs, args.distant_tsm_model_base,
            distant_tsm_roots),
    }

    val_cache = {}
    bundle_cache = {}
    plot_base = os.path.join(project_root, args.save_plot) if args.save_plot else None
    plot_pdf_base = os.path.join(project_root, args.save_plot_pdf) if args.save_plot_pdf else None
    summary_base = os.path.join(project_root, args.save_summary) if args.save_summary else None

    for method, pairs in method_pairs.items():
        if not pairs:
            continue
        records = []
        for stub, (local_path, distant_path) in pairs.items():
            local_cfg = _load_saved_config(local_path)
            distant_cfg = _load_saved_config(distant_path)
            if method == "tdre":
                _ensure_tdre_checkpoint_dir(local_cfg)
                _ensure_tdre_checkpoint_dir(distant_cfg)
            local_entry = _get_val_data(local_cfg, val_cache, f"{stub}_local")
            distant_entry = _get_val_data(distant_cfg, val_cache, f"{stub}_dist")
            local_info = {
                "path": local_path,
                "config": local_cfg,
                "val_entry": local_entry,
            }
            distant_info = {
                "path": distant_path,
                "config": distant_cfg,
                "val_entry": distant_entry,
            }

            bundle = build_bundle_from_configs(
                stub,
                local_cfg,
                distant_cfg,
                args.eval_size,
                args.n_eval_trials,
                args.seed_offset,
                bundle_cache,
            )

            if requested_shifts and all(
                abs(bundle["kl_shift"] - target) > shift_tol for target in requested_shifts
            ):
                continue

            est_local, est_distant, est_r, local_trials, distant_trials = estimate_method(
                method,
                local_info,
                distant_info,
                bundle["samples"],
            )
            abs_err = abs(est_r - bundle["true_r"])
            rel_err = abs_err / max(abs(bundle["true_r"]), 1e-8)
            trial_estimates_p0 = [float(v) for v in distant_trials]
            trial_rel_errors = []
            for l_val, d_val in zip(local_trials, distant_trials):
                est_delta = float(l_val - d_val)
                trial_rel_errors.append(
                    abs(est_delta - bundle["true_r"]) / max(abs(bundle["true_r"]), 1e-8)
                )
            print(
                f"[{method.upper()}][{stub}] shift={bundle['kl_shift']:.4f} | "
                f"Est KL(P*||P1)={est_local:.4f} (true {bundle['kl_pstar_p1']:.4f}), "
                f"Est KL(P*||P0)={est_distant:.4f} (true {bundle['kl_pstar_p0']:.4f}), "
                f"Est Δ={est_r:.4f}, True Δ={bundle['true_r']:.4f}"
            )
            records.append(
                {
                    "method": method,
                    "stub": stub,
                    "kl_shift": bundle["kl_shift"],
                    "kl_p0_p1": bundle["kl_p0_p1"],
                    "kl_p0_pstar": bundle["kl_p0_pstar"],
                    "kl_pstar_p1": bundle["kl_pstar_p1"],
                    "kl_pstar_p0": bundle["kl_pstar_p0"],
                    "true_r": bundle["true_r"],
                    "est_kl_pstar_p1": est_local,
                    "est_kl_pstar_p0": est_distant,
                    "est_r": est_r,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "trial_estimates_p0": trial_estimates_p0,
                    "trial_rel_errors": trial_rel_errors,
                }
            )

        if not records:
            continue

        unique_shifts = sorted({round(rec["kl_shift"], 5) for rec in records})
        for shift in unique_shifts:
            png_path = format_output_path(plot_base, f"{method}_average_err", shift)
            pdf_path = format_output_path(plot_pdf_base, f"{method}_average_err", shift)
            plot_method(records, method, shift, png_path, pdf_path)
            png_trials = format_output_path(plot_base, f"{method}_scatter_err", shift)
            pdf_trials = format_output_path(plot_pdf_base, f"{method}_scatter_err", shift)
            plot_estimated_trials(records, method, shift, png_trials, pdf_trials)

        summary_path = format_output_path(summary_base, f"{method}_kl", None)
        if summary_path:
            save_summary(method, records, unique_shifts, summary_path)


if __name__ == "__main__":
    main()
