"""Compare MultinomialTRE, MultiheadTRE, and DirectDRE on toy Gaussian problem.

Ground truth: For two multivariate Gaussians p = N(mu_p, Sigma_p) and q = N(mu_q, Sigma_q),
the log density ratio at x is:

log p(x)/q(x) = -0.5 * [
    log|Sigma_p|/|Sigma_q|
    + (x - mu_p)^T Sigma_p^{-1} (x - mu_p)
    - (x - mu_q)^T Sigma_q^{-1} (x - mu_q)
]
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.fragments.dre.direct import DirectDRE, DirectDREConfig
from src.fragments.dre.telescoping import MultiheadTRE, MultiheadTREConfig, MultinomialTRE, MultinomialTREConfig


def compute_ground_truth_log_ratio(
    x: torch.Tensor,
    mu_p: torch.Tensor,
    cov_p: torch.Tensor,
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
) -> torch.Tensor:
    """Compute closed-form log density ratio for Gaussians.

    Args:
        x: Points to evaluate, shape (batch_size, dim)
        mu_p: Mean of numerator distribution, shape (dim,)
        cov_p: Covariance of numerator distribution, shape (dim, dim)
        mu_q: Mean of denominator distribution, shape (dim,)
        cov_q: Covariance of denominator distribution, shape (dim, dim)

    Returns:
        Log density ratios, shape (batch_size,)
    """
    dim = x.shape[1]

    # Log determinant terms
    log_det_p = torch.linalg.slogdet(cov_p)[1]
    log_det_q = torch.linalg.slogdet(cov_q)[1]

    # Precision matrices
    prec_p = torch.linalg.inv(cov_p)
    prec_q = torch.linalg.inv(cov_q)

    # Mahalanobis distances
    diff_p = x - mu_p.unsqueeze(0)  # (batch, dim)
    diff_q = x - mu_q.unsqueeze(0)  # (batch, dim)

    mahal_p = torch.sum(diff_p @ prec_p * diff_p, dim=1)  # (batch,)
    mahal_q = torch.sum(diff_q @ prec_q * diff_q, dim=1)  # (batch,)

    # Log ratio: log p(x) - log q(x)
    # = -0.5 * (log|Sigma_p| + mahal_p) + 0.5 * (log|Sigma_q| + mahal_q)
    # = 0.5 * (log|Sigma_q| - log|Sigma_p| + mahal_q - mahal_p)
    log_ratio = 0.5 * (log_det_q - log_det_p + mahal_q - mahal_p)

    return log_ratio


def generate_gaussian_data(
    n_samples: int,
    mu_p: torch.Tensor,
    cov_p: torch.Tensor,
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples from two Gaussian distributions.

    Returns:
        Tuple of (samples_p, samples_q), each shape (n_samples, dim)
    """
    dim = mu_p.shape[0]

    # Create distributions
    dist_p = torch.distributions.MultivariateNormal(mu_p, cov_p)
    dist_q = torch.distributions.MultivariateNormal(mu_q, cov_q)

    # Sample
    samples_p = dist_p.sample((n_samples,)).to(device)
    samples_q = dist_q.sample((n_samples,)).to(device)

    return samples_p, samples_q


@dataclass
class ExperimentConfig:
    """Configuration for the comparison experiment."""
    var_dim: int = 20
    n_train_samples: int = 10000
    n_test_samples: int = 2000
    num_steps_list: Tuple[int, ...] = (2, 4, 8)  # Different m values to test
    base_hidden_dim: int = 64  # Hidden dim for single head / DirectDRE
    num_blocks: int = 3
    block_depth: int = 2
    epochs: int = 500
    batch_size: int = 256
    lr: float = 3e-4
    patience: int = 30
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    # Distribution parameters
    mean_shift: float = 3.0  # Shift in mean
    var_p: float = 0.3  # Variance of p (numerator)
    var_q: float = 2.0  # Variance of q (denominator)


def run_experiment(config: ExperimentConfig):
    """Run comparison experiment."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Running on device: {config.device}")
    print(f"Variable dimension: {config.var_dim}")
    print(f"Telescoping steps to test (m): {config.num_steps_list}")
    print(f"Base hidden dim (single head / DirectDRE): {config.base_hidden_dim}")
    print()

    # Setup Gaussian distributions
    mu_p = torch.zeros(config.var_dim, device=config.device)
    mu_q = torch.ones(config.var_dim, device=config.device) * config.mean_shift

    cov_p = torch.eye(config.var_dim, device=config.device) * config.var_p
    cov_q = torch.eye(config.var_dim, device=config.device) * config.var_q

    print("Distribution setup:")
    print(f"  p ~ N(0, {config.var_p}*I)")
    print(f"  q ~ N({config.mean_shift}, {config.var_q}*I)")
    print()

    # Generate data
    print("Generating data...")
    train_p, train_q = generate_gaussian_data(
        config.n_train_samples, mu_p, cov_p, mu_q, cov_q, config.device
    )
    test_p, test_q = generate_gaussian_data(
        config.n_test_samples, mu_p, cov_p, mu_q, cov_q, config.device
    )

    # Compute ground truth on test samples from both distributions
    test_all = torch.cat([test_p, test_q], dim=0)
    ground_truth = compute_ground_truth_log_ratio(test_all, mu_p, cov_p, mu_q, cov_q)

    print(f"Ground truth log ratio stats:")
    print(f"  On p samples: mean={ground_truth[:config.n_test_samples].mean():.4f}, "
          f"std={ground_truth[:config.n_test_samples].std():.4f}")
    print(f"  On q samples: mean={ground_truth[config.n_test_samples:].mean():.4f}, "
          f"std={ground_truth[config.n_test_samples:].std():.4f}")
    print()

    def correlation(pred, target):
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        return (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm()
        )

    all_results = {}

    # =========================================================================
    # 1. DirectDRE (baseline, single head hidden dim)
    # =========================================================================
    print("=" * 70)
    print("Training DirectDRE (baseline)...")
    print(f"  Hidden dim: {config.base_hidden_dim}")

    direct_config = DirectDREConfig(
        var_dim=config.var_dim,
        loss='nce',
        nn_hidden_dim=config.base_hidden_dim,
        nn_num_blocks=config.num_blocks,
        nn_block_depth=config.block_depth,
        nn_norm='none',
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        patience=config.patience,
        device=config.device,
    )

    direct_dre = DirectDRE(direct_config)
    direct_dre.fit(train_p, train_q)

    with torch.no_grad():
        direct_pred = direct_dre(test_all)

    direct_mse = ((direct_pred - ground_truth) ** 2).mean().item()
    direct_mae = (direct_pred - ground_truth).abs().mean().item()
    direct_bias = (direct_pred - ground_truth).mean().item()
    direct_corr = correlation(direct_pred, ground_truth).item()

    all_results['DirectDRE'] = {
        'mse': direct_mse,
        'mae': direct_mae,
        'bias': direct_bias,
        'corr': direct_corr,
        'hidden_dim': config.base_hidden_dim,
    }

    print(f"  MSE: {direct_mse:.4f}, MAE: {direct_mae:.4f}, Bias: {direct_bias:.4f}, Corr: {direct_corr:.4f}")
    print()

    # =========================================================================
    # Loop over different m values
    # =========================================================================
    for num_steps in config.num_steps_list:
        print("=" * 70)
        print(f"Testing with m = {num_steps} telescoping steps")
        print("=" * 70)

        # Reset seed for fair comparison
        torch.manual_seed(config.seed + num_steps)

        # ---------------------------------------------------------------------
        # MultiheadTRE
        # ---------------------------------------------------------------------
        print(f"\nTraining MultiheadTRE (m={num_steps})...")
        print(f"  Hidden dim per head: {config.base_hidden_dim}, Num heads: {num_steps}")

        multihead_config = MultiheadTREConfig(
            var_dim=config.var_dim,
            num_steps=num_steps,
            loss='nce',
            nn_hidden_dim=config.base_hidden_dim,
            nn_num_blocks=config.num_blocks,
            nn_block_depth=config.block_depth,
            nn_norm='none',
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            patience=config.patience,
            device=config.device,
        )

        multihead_tre = MultiheadTRE(multihead_config)
        multihead_tre.fit(train_p, train_q)

        with torch.no_grad():
            multihead_pred = multihead_tre(test_all)

        multihead_mse = ((multihead_pred - ground_truth) ** 2).mean().item()
        multihead_mae = (multihead_pred - ground_truth).abs().mean().item()
        multihead_bias = (multihead_pred - ground_truth).mean().item()
        multihead_corr = correlation(multihead_pred, ground_truth).item()

        all_results[f'MultiheadTRE_m{num_steps}'] = {
            'mse': multihead_mse,
            'mae': multihead_mae,
            'bias': multihead_bias,
            'corr': multihead_corr,
            'hidden_dim': config.base_hidden_dim,
            'num_heads': num_steps,
        }

        print(f"  MSE: {multihead_mse:.4f}, MAE: {multihead_mae:.4f}, Bias: {multihead_bias:.4f}, Corr: {multihead_corr:.4f}")

        # ---------------------------------------------------------------------
        # MultinomialTRE
        # ---------------------------------------------------------------------
        multinomial_hidden = num_steps * config.base_hidden_dim
        print(f"\nTraining MultinomialTRE (m={num_steps})...")
        print(f"  Hidden dim: {multinomial_hidden} (= {num_steps} * {config.base_hidden_dim})")

        multinomial_config = MultinomialTREConfig(
            var_dim=config.var_dim,
            num_steps=num_steps,
            nn_hidden_dim=multinomial_hidden,
            nn_num_blocks=config.num_blocks,
            nn_block_depth=config.block_depth,
            nn_norm='none',
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            patience=config.patience,
            device=config.device,
        )

        multinomial_tre = MultinomialTRE(multinomial_config)
        multinomial_tre.fit(train_p, train_q)

        with torch.no_grad():
            multinomial_pred = multinomial_tre(test_all)

        multinomial_mse = ((multinomial_pred - ground_truth) ** 2).mean().item()
        multinomial_mae = (multinomial_pred - ground_truth).abs().mean().item()
        multinomial_bias = (multinomial_pred - ground_truth).mean().item()
        multinomial_corr = correlation(multinomial_pred, ground_truth).item()

        all_results[f'MultinomialTRE_m{num_steps}'] = {
            'mse': multinomial_mse,
            'mae': multinomial_mae,
            'bias': multinomial_bias,
            'corr': multinomial_corr,
            'hidden_dim': multinomial_hidden,
        }

        print(f"  MSE: {multinomial_mse:.4f}, MAE: {multinomial_mae:.4f}, Bias: {multinomial_bias:.4f}, Corr: {multinomial_corr:.4f}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Method':<25} {'Hidden Dim':<15} {'MSE':<12} {'MAE':<10} {'Bias':<12} {'Corr':<8}")
    print("-" * 90)

    base_mse = all_results['DirectDRE']['mse']

    for name, res in all_results.items():
        hd = res['hidden_dim']
        if 'num_heads' in res:
            hd = f"{hd} x {res['num_heads']}"
        ratio = res['mse'] / base_mse
        print(f"{name:<25} {str(hd):<15} {res['mse']:<12.4f} {res['mae']:<10.4f} {res['bias']:<12.4f} {res['corr']:<8.4f} ({ratio:.2f}x)")

    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    results = run_experiment(config)
