"""Aggregate results from parallel DRE experiments and create plots.

Usage:
    python -m src.unit_tests.aggregate_dre_results \
        --hd_nats_dir <path> \
        --hnats_d_dir <path> \
        --hnatsd_samples_dir <path>
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime

import torch
import matplotlib.pyplot as plt


def load_experiment_results(input_dir: Path, experiment_type: str):
    """Load all pickle files for an experiment type."""
    if not input_dir.exists():
        print(f"Warning: Directory not found: {input_dir}")
        return {}

    pkl_files = sorted(input_dir.glob(f"{experiment_type}_*.pkl"))

    if not pkl_files:
        print(f"Warning: No result files found in {input_dir}")
        return {}

    print(f"Loading {len(pkl_files)} files from {input_dir}")

    results_by_param = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            param_value = data['param_value']
            results_by_param[param_value] = data['results']

    return results_by_param


def create_plots(param_name, param_values, all_results, output_dir):
    """Create comprehensive plots for DRE experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = list(all_results[0].keys())

    # Plot 1: MSE vs Parameter (Test and OOD)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for i, method_name in enumerate(method_names):
        mse_test_values = [results[method_name]['mse_test'] for results in all_results]
        mse_ood_values = [results[method_name]['mse_ood'] for results in all_results]

        color = f'C{i % 10}'
        ax1.plot(param_values, mse_test_values, marker='o', label=method_name,
                 alpha=0.7, linewidth=2, color=color)
        ax2.plot(param_values, mse_ood_values, marker='o', label=method_name,
                 alpha=0.7, linewidth=2, color=color)

    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Test MSE')
    ax1.set_title(f'Test MSE vs {param_name}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel(param_name)
    ax2.set_ylabel('OOD MSE')
    ax2.set_title(f'OOD MSE vs {param_name}')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'mse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir / 'mse_comparison.png'}")

    # Plot 2: Scatter plot of true vs estimated log ratios (using last parameter value)
    last_results = all_results[-1]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'P']

    for i, method_name in enumerate(method_names):
        test_gt = last_results[method_name]['test_gt_log_ratios'].cpu().numpy()
        test_pred = last_results[method_name]['test_pred_log_ratios'].cpu().numpy()

        n_samples = min(1000, len(test_gt))
        indices = torch.randperm(len(test_gt))[:n_samples].numpy()

        marker = markers[i % len(markers)]
        color = f'C{i % 10}'
        ax1.scatter(test_gt[indices], test_pred[indices], marker=marker,
                   label=method_name, alpha=0.5, s=30, color=color)

        ood_gt = last_results[method_name]['ood_gt_log_ratios'].cpu().numpy()
        ood_pred = last_results[method_name]['ood_pred_log_ratios'].cpu().numpy()

        indices_ood = torch.randperm(len(ood_gt))[:n_samples].numpy()
        ax2.scatter(ood_gt[indices_ood], ood_pred[indices_ood], marker=marker,
                   label=method_name, alpha=0.5, s=30, color=color)

    all_test_gt = torch.cat([last_results[m]['test_gt_log_ratios'] for m in method_names]).cpu().numpy()
    test_min, test_max = all_test_gt.min(), all_test_gt.max()
    ax1.plot([test_min, test_max], [test_min, test_max], 'k--', alpha=0.5, linewidth=1.5, label='Perfect fit')

    all_ood_gt = torch.cat([last_results[m]['ood_gt_log_ratios'] for m in method_names]).cpu().numpy()
    ood_min, ood_max = all_ood_gt.min(), all_ood_gt.max()
    ax2.plot([ood_min, ood_max], [ood_min, ood_max], 'k--', alpha=0.5, linewidth=1.5, label='Perfect fit')

    ax1.set_xlabel('True Log Ratio')
    ax1.set_ylabel('Estimated Log Ratio')
    ax1.set_title(f'Test Set: True vs Estimated ({param_name}={param_values[-1]})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('True Log Ratio')
    ax2.set_ylabel('Estimated Log Ratio')
    ax2.set_title(f'OOD Set: True vs Estimated ({param_name}={param_values[-1]})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'log_ratio_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / 'log_ratio_scatter.png'}")

    # Plot 3: Error by distance (using last parameter value)
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, method_name in enumerate(method_names):
        errors_p = last_results[method_name]['errors_by_dist_p'].cpu().numpy()
        errors_q = last_results[method_name]['errors_by_dist_q'].cpu().numpy()

        n_samples = min(1000, len(errors_p))
        indices = torch.randperm(len(errors_p))[:n_samples].numpy()

        color = f'C{i % 10}'

        ax1.scatter(errors_p[indices, 0], errors_p[indices, 1],
                   alpha=0.3, s=20, color=color, label=method_name)

        ax2.scatter(errors_q[indices, 0], errors_q[indices, 1],
                   alpha=0.3, s=20, color=color, label=method_name)

    ax1.set_xlabel('Distance to μ_p')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title(f'Error vs Distance to μ_p ({param_name}={param_values[-1]})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel('Distance to μ_q')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(f'Error vs Distance to μ_q ({param_name}={param_values[-1]})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_distance.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir / 'error_by_distance.png'}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate parallel DRE results')
    parser.add_argument('--hd_nats_dir', type=str, default=None,
                        help='Directory with hd_nats results')
    parser.add_argument('--hnats_d_dir', type=str, default=None,
                        help='Directory with hnats_d results')
    parser.add_argument('--hnatsd_samples_dir', type=str, default=None,
                        help='Directory with hnatsd_samples results')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process hd_nats
    if args.hd_nats_dir:
        print("\n" + "="*80)
        print("Processing hd_nats...")
        print("="*80)
        results = load_experiment_results(Path(args.hd_nats_dir), 'hd_nats')
        if results:
            nats_values = sorted(results.keys())
            results_list = [results[v] for v in nats_values]
            output_dir = Path(args.hd_nats_dir) / 'plots' / timestamp
            create_plots('KL (nats)', nats_values, results_list, output_dir)

    # Process hnats_d
    if args.hnats_d_dir:
        print("\n" + "="*80)
        print("Processing hnats_d...")
        print("="*80)
        results = load_experiment_results(Path(args.hnats_d_dir), 'hnats_d')
        if results:
            d_values = sorted(results.keys())
            results_list = [results[v] for v in d_values]
            output_dir = Path(args.hnats_d_dir) / 'plots' / timestamp
            create_plots('Dimension', d_values, results_list, output_dir)

    # Process hnatsd_samples
    if args.hnatsd_samples_dir:
        print("\n" + "="*80)
        print("Processing hnatsd_samples...")
        print("="*80)
        results = load_experiment_results(Path(args.hnatsd_samples_dir), 'hnatsd_samples')
        if results:
            n_values = sorted(results.keys())
            results_list = [results[v] for v in n_values]
            output_dir = Path(args.hnatsd_samples_dir) / 'plots' / timestamp
            create_plots('Sample size', n_values, results_list, output_dir)

    print(f"\n{'='*80}")
    print(f"✓ All plots saved")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
