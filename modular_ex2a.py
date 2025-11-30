"""Parallel DRE experiment runner using multiprocessing within each task.

Usage:
    python -m src.unit_tests.dre_plot_test_parallel \
        --experiment_type <hd_nats|hnats_d|hnatsd_samples> \
        --param_value <value> \
        --output_dir <path> \
        --num_gpus <N>
"""

import argparse
import pickle
from pathlib import Path
import torch
from joblib import Parallel, delayed
from functools import partial

from ..methods.vbn.gauss.gaussian_ops import exact_kl
from ..fragments.dre.direct import DirectDRE, DirectDREConfig
from ..fragments.dre.telescoping import MultinomialTRE, MultinomialTREConfig, MultiheadTRE, MultiheadTREConfig
from torch.distributions import MultivariateNormal


def compute_log_ratio_batch(x, mu_p, mu_q, prec):
    """Vectorized ground truth log ratio computation."""
    diff_p = x - mu_p
    diff_q = x - mu_q
    mahal_p = torch.sum(diff_p @ prec * diff_p, dim=1)
    mahal_q = torch.sum(diff_q @ prec * diff_q, dim=1)
    return 0.5 * (mahal_q - mahal_p)


def craft_normals(dim: int, target_kl: float, device: str):
    """Create pair of Gaussian distributions with specified KL divergence."""
    mu_p = torch.zeros(dim, device=device)

    # Use diagonal covariance for numerical stability
    # Scale each dimension's variance by a small random factor around 1.0
    variances = torch.ones(dim, device=device) + 0.5 * torch.randn(dim, device=device).abs()
    cov = torch.diag(variances)

    dist_p = MultivariateNormal(loc=mu_p, covariance_matrix=cov)

    # Compute mean shift to achieve target KL divergence
    # For same covariance: KL(p||q) = 0.5 * (μ_q - μ_p)^T Σ^{-1} (μ_q - μ_p)
    # We want KL = target_kl, so: ||Σ^{-0.5}(μ_q - μ_p)||^2 = 2 * target_kl
    # Let Σ^{-0.5}(μ_q - μ_p) = [sqrt(2*target_kl), 0, 0, ...]
    scaling_factor = torch.sqrt(torch.tensor(2.0 * target_kl, device=device))
    z = torch.zeros(dim, device=device)
    z[0] = scaling_factor
    # delta = Σ^{0.5} @ z
    delta = torch.sqrt(variances) * z
    mu_q = mu_p + delta
    dist_q = MultivariateNormal(loc=mu_q, covariance_matrix=cov)

    return dist_p, dist_q


def train_single_method(method_idx, method_name, d, nats, n, shared_data, gpu_id):
    """Train a single DRE method on assigned GPU.

    Args:
        method_idx: Index of this method (for identification)
        method_name: Name of the method
        d: Dimension
        nats: KL divergence
        n: Sample size
        shared_data: Dict with shared tensors (samples, ground truth, etc.)
        gpu_id: GPU ID to use
    """
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'

    print(f"[GPU {gpu_id}] Training {method_name}...")

    # Move shared data to this GPU
    train_p = shared_data['train_p'].to(device)
    train_q = shared_data['train_q'].to(device)
    test_samples = shared_data['test_samples'].to(device)
    test_p = shared_data['test_p'].to(device)
    ood_samples = shared_data['ood_samples'].to(device)
    test_gt_log_ratios = shared_data['test_gt_log_ratios'].to(device)
    ood_gt_log_ratios = shared_data['ood_gt_log_ratios'].to(device)
    dist_to_mu_p = shared_data['dist_to_mu_p'].to(device)
    dist_to_mu_q = shared_data['dist_to_mu_q'].to(device)
    ood_gt_expected = shared_data['ood_gt_expected']

    # Create config based on method name (matching test_tre_comparison.py)
    if method_name == 'DirectDRE':
        config = DirectDREConfig(
            var_dim=d,
            loss='nce',
            nn_hidden_dim=32,
            nn_num_blocks=3,
            nn_block_depth=2,
            nn_norm='none',
            epochs=500,
            batch_size=256,
            lr=3e-4,
            patience=30,
            device=device,
        )
        dre = DirectDRE(config)
    elif method_name.startswith('MultinomialTRE_'):
        m = int(method_name.split('_')[1])
        config = MultinomialTREConfig(
            var_dim=d,
            num_steps=m,
            nn_hidden_dim=32,
            nn_num_blocks=3,
            nn_block_depth=2,
            nn_norm='none',
            epochs=500,
            batch_size=256,
            lr=3e-4,
            patience=30,
            device=device,
        )
        dre = MultinomialTRE(config)
    elif method_name.startswith('MultiheadTRE_'):
        m = int(method_name.split('_')[1])
        config = MultiheadTREConfig(
            var_dim=d,
            num_steps=m,
            loss='nce',
            nn_hidden_dim=32,
            nn_num_blocks=3,
            nn_block_depth=2,
            nn_norm='none',
            epochs=500,
            batch_size=256,
            lr=3e-4,
            patience=30,
            device=device,
        )
        dre = MultiheadTRE(config)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Train
    dre.fit(train_p, train_q)

    # Evaluate
    with torch.no_grad():
        test_pred_log_ratios = dre(test_samples)
        kl_estimated = dre(test_p).mean().item()
        ood_pred_log_ratios = dre(ood_samples)

    mse_test = ((test_pred_log_ratios - test_gt_log_ratios) ** 2).mean().item()
    mse_ood = ((ood_pred_log_ratios - ood_gt_log_ratios) ** 2).mean().item()
    ood_pred_expected = ood_pred_log_ratios.mean().item()

    abs_errors = torch.abs(test_pred_log_ratios - test_gt_log_ratios)
    errors_by_dist_p = torch.stack([dist_to_mu_p, abs_errors], dim=1)
    errors_by_dist_q = torch.stack([dist_to_mu_q, abs_errors], dim=1)

    results = {
        'mse_test': mse_test,
        'mse_ood': mse_ood,
        'kl_estimated': kl_estimated,
        'ood_pred': ood_pred_expected,
        'ood_true': ood_gt_expected,
        'errors_by_dist_p': errors_by_dist_p.cpu(),
        'errors_by_dist_q': errors_by_dist_q.cpu(),
        'test_pred_log_ratios': test_pred_log_ratios.cpu(),
        'test_gt_log_ratios': test_gt_log_ratios.cpu(),
        'ood_pred_log_ratios': ood_pred_log_ratios.cpu(),
        'ood_gt_log_ratios': ood_gt_log_ratios.cpu(),
    }

    print(f"[GPU {gpu_id}] {method_name} complete: Test MSE={mse_test:.6f}, OOD MSE={mse_ood:.6f}")

    return method_name, results


def run_experiment(experiment_type: str, param_value, num_gpus: int):
    """Run all methods for a single parameter value using multiprocessing."""

    # Map experiment type to parameters (ensure correct types)
    if experiment_type == 'hd_nats':
        d, nats, n = 32, float(param_value), 32768
    elif experiment_type == 'hnats_d':
        d, nats, n = int(param_value), 10.0, 32768
    elif experiment_type == 'hnatsd_samples':
        d, nats, n = 32, 10.0, int(param_value)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    print(f"\n{'='*80}")
    print(f"Running {experiment_type}: d={d}, nats={nats}, n={n}")
    print(f"Using {num_gpus} GPUs")
    print(f"{'='*80}\n")

    # Create distributions and sample data on CPU (will be moved to GPUs later)
    device = 'cpu'
    p, q = craft_normals(d, nats, device)

    train_p = p.sample((n,))
    train_q = q.sample((n,))

    test_p = p.sample((n // 2,))
    test_q = q.sample((n // 2,))
    test_samples = torch.cat([test_p, test_q], dim=0)
    test_perm = torch.randperm(test_samples.shape[0], device=device)
    test_samples = test_samples[test_perm]

    midpoint = (p.mean + q.mean) / 2
    ood_dist = torch.distributions.Laplace(loc=midpoint, scale=nats)
    ood_samples = ood_dist.sample((n,))

    # Compute ground truth
    mu_p = p.mean
    mu_q = q.mean
    cov = p.covariance_matrix
    prec = torch.linalg.inv(cov)

    test_gt_log_ratios = compute_log_ratio_batch(test_samples, mu_p, mu_q, prec)
    ood_gt_log_ratios = compute_log_ratio_batch(ood_samples, mu_p, mu_q, prec)
    ood_gt_expected = ood_gt_log_ratios.mean().item()

    dist_to_mu_p = torch.norm(test_samples - mu_p, dim=1)
    dist_to_mu_q = torch.norm(test_samples - mu_q, dim=1)

    # Package shared data
    shared_data = {
        'train_p': train_p,
        'train_q': train_q,
        'test_samples': test_samples,
        'test_p': test_p,
        'ood_samples': ood_samples,
        'test_gt_log_ratios': test_gt_log_ratios,
        'ood_gt_log_ratios': ood_gt_log_ratios,
        'ood_gt_expected': ood_gt_expected,
        'dist_to_mu_p': dist_to_mu_p,
        'dist_to_mu_q': dist_to_mu_q,
    }

    # Define methods
    methods = [
        'DirectDRE',
        'MultinomialTRE_2', 'MultinomialTRE_8', 'MultinomialTRE_16',
        'MultiheadTRE_2', 'MultiheadTRE_8', 'MultiheadTRE_16'
    ]

    # Assign GPUs to methods (round-robin)
    if num_gpus > 0:
        gpu_assignments = [i % num_gpus for i in range(len(methods))]
    else:
        gpu_assignments = [None] * len(methods)

    print(f"GPU assignments: {dict(zip(methods, gpu_assignments))}\n")

    # Use joblib with loky backend to train methods in parallel
    n_jobs = min(len(methods), num_gpus if num_gpus > 0 else len(methods))

    results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(train_single_method)(i, method, d, nats, n, shared_data, gpu_id)
        for i, (method, gpu_id) in enumerate(zip(methods, gpu_assignments))
    )

    # Convert to dict
    results = {method_name: res for method_name, res in results_list}

    return results


def main():
    parser = argparse.ArgumentParser(description='Parallel DRE experiments')
    parser.add_argument('--experiment_type', type=str, required=True,
                        choices=['hd_nats', 'hnats_d', 'hnatsd_samples'],
                        help='Type of experiment')
    parser.add_argument('--param_value', type=float, required=True,
                        help='Parameter value for this task')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs available')

    args = parser.parse_args()

    # Run experiment
    results = run_experiment(args.experiment_type, args.param_value, args.num_gpus)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.experiment_type}_{args.param_value}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'experiment_type': args.experiment_type,
            'param_value': args.param_value,
            'results': results,
        }, f)

    print(f"\n✓ Saved results to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  Test MSE: {metrics['mse_test']:.6f}")
        print(f"  OOD MSE: {metrics['mse_ood']:.6f}")
        print(f"  KL (estimated): {metrics['kl_estimated']:.6f}")
        print(f"  OOD E[log ratio] (true): {metrics['ood_true']:.6f}, (estimated): {metrics['ood_pred']:.6f}")


if __name__ == '__main__':
    main()
