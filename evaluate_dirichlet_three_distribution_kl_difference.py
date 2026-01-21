import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors

tf_disable = False
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf_disable = True
except Exception:
    tf_disable = False

from evaluate_gaussians_10d import (
    _collect_model_dirs,
    _get_val_data,
    _load_saved_config,
)
from evaluate_gaussians_three_distribution import (
    bdre_estimates,
    dv_estimates,
    nwj_estimates,
    collect_method_dirs,
    format_output_path,
)
from evaluate_gaussians_three_distribution_kl_difference import (
    build_bundle_from_configs,
    estimate_method,
    plot_estimated_trials,
    plot_method,
    save_summary,
    gather_method_pairs,
)
from utils.experiment_utils import project_root


KL_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]

DIRICHLET_METHOD_ROOTS = {
    "dv": ("dv_dirichlet_pstar_p1_kl{}", "dv_dirichlet_pstar_p0_kl{}"),
    "nwj": ("nwj_dirichlet_pstar_p1_kl{}", "nwj_dirichlet_pstar_p0_kl{}"),
    "bdre": ("bdre_dirichlet_pstar_p1_kl{}", "bdre_dirichlet_pstar_p0_kl{}"),
}


def _default_roots(method, is_local):
    if method not in DIRICHLET_METHOD_ROOTS:
        raise ValueError(f"Unsupported method {method}")
    local_fmt, distant_fmt = DIRICHLET_METHOD_ROOTS[method]
    fmt = local_fmt if is_local else distant_fmt
    return [fmt.format(kl) for kl in KL_VALUES]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Dirichlet three-distribution KL differences for DV/NWJ/BDRE."
    )
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--n_eval_trials", type=int, default=20)
    parser.add_argument("--kl_shift", type=float, default=0.5)
    parser.add_argument("--kl_shifts", type=float, nargs="+", default=None)
    parser.add_argument("--seed_offset", type=int, default=98765)
    parser.add_argument("--shift_tol", type=float, default=1e-3,
                        help="Tolerance when matching KL(P1||P*) shifts")

    parser.add_argument("--local_dv_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_dv_model_base", type=str, default=None)
    parser.add_argument("--distant_dv_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_dv_model_base", type=str, default=None)
    parser.add_argument("--local_dv_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_dv_model_roots", nargs="+", default=None)

    parser.add_argument("--local_nwj_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_nwj_model_base", type=str, default=None)
    parser.add_argument("--distant_nwj_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_nwj_model_base", type=str, default=None)
    parser.add_argument("--local_nwj_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_nwj_model_roots", nargs="+", default=None)

    parser.add_argument("--local_bdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--local_bdre_model_base", type=str, default=None)
    parser.add_argument("--distant_bdre_model_dirs", nargs="+", default=None)
    parser.add_argument("--distant_bdre_model_base", type=str, default=None)
    parser.add_argument("--local_bdre_model_roots", nargs="+", default=None)
    parser.add_argument("--distant_bdre_model_roots", nargs="+", default=None)

    parser.add_argument("--save_plot", type=str, default="results/dirichlet_three_dist/plot.png")
    parser.add_argument("--save_plot_pdf", type=str, default="results/dirichlet_three_dist/plot.pdf")
    parser.add_argument("--save_summary", type=str, default="results/dirichlet_three_dist/summary.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_shifts = args.kl_shifts if args.kl_shifts else [args.kl_shift]
    requested_shifts = [float(s) for s in requested_shifts]
    shift_tol = float(args.shift_tol)

    local_dv_roots = args.local_dv_model_roots or _default_roots("dv", True)
    distant_dv_roots = args.distant_dv_model_roots or _default_roots("dv", False)
    local_nwj_roots = args.local_nwj_model_roots or _default_roots("nwj", True)
    distant_nwj_roots = args.distant_nwj_model_roots or _default_roots("nwj", False)
    local_bdre_roots = args.local_bdre_model_roots or _default_roots("bdre", True)
    distant_bdre_roots = args.distant_bdre_model_roots or _default_roots("bdre", False)

    method_pairs = {
        "dv": gather_method_pairs(
            args.local_dv_model_dirs,
            args.local_dv_model_base,
            local_dv_roots,
            args.distant_dv_model_dirs,
            args.distant_dv_model_base,
            distant_dv_roots,
        ),
        "nwj": gather_method_pairs(
            args.local_nwj_model_dirs,
            args.local_nwj_model_base,
            local_nwj_roots,
            args.distant_nwj_model_dirs,
            args.distant_nwj_model_base,
            distant_nwj_roots,
        ),
        "bdre": gather_method_pairs(
            args.local_bdre_model_dirs,
            args.local_bdre_model_base,
            local_bdre_roots,
            args.distant_bdre_model_dirs,
            args.distant_bdre_model_base,
            distant_bdre_roots,
        ),
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
            local_entry = _get_val_data(local_cfg, val_cache, f"{stub}_local")
            distant_entry = _get_val_data(distant_cfg, val_cache, f"{stub}_dist")
            local_info = {"path": local_path, "config": local_cfg, "val_entry": local_entry}
            distant_info = {"path": distant_path, "config": distant_cfg, "val_entry": distant_entry}

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
                method, local_info, distant_info, bundle["samples"]
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
        for shift in requested_shifts:
            png_path = format_output_path(plot_base, f"{method}_average_err", shift)
            pdf_path = format_output_path(plot_pdf_base, f"{method}_average_err", shift)
            plot_method(records, method, shift, png_path, pdf_path, shift_tol=shift_tol)
            png_trials = format_output_path(plot_base, f"{method}_scatter_err", shift)
            pdf_trials = format_output_path(plot_pdf_base, f"{method}_scatter_err", shift)
            plot_estimated_trials(records, method, shift, png_trials, pdf_trials, shift_tol=shift_tol)

        summary_path = format_output_path(summary_base, f"{method}_kl", None)
        if summary_path:
            save_summary(method, records, unique_shifts, summary_path)


if __name__ == "__main__":
    main()
