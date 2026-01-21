"""
Multivariate Gaussian Experiment for TDRE Evaluation

This script evaluates TDRE performance on multivariate Gaussian distributions with 
different KL divergences (10, 15, 20 nats) and training sample sizes (50, 100, 300).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from argparse import ArgumentParser
from data_handlers.gaussians import GAUSSIANS
from utils.misc_utils import kl_between_two_gaussians, AttrDict
from build_bridges import build_graph as build_tdre_graph
import json
import logging


def create_gaussian_distributions_by_kl(target_kl, n_dims=10):
    """
    Create two multivariate Gaussian distributions with a target KL divergence.
    
    For P1 = N(mu1, I) and P0 = N(0, I), we have:
    KL(P1||P0) = 0.5 * ||mu1||^2
    
    So we set mu1 = sqrt(2*KL) / sqrt(n_dims) * ones(n_dims)
    
    Args:
        target_kl: Target KL divergence (nats)
        n_dims: Dimensionality
        
    Returns:
        numerator_mean, numerator_cov, denominator_mean, denominator_cov, actual_kl
    """
    # P0 (denominator): N(0, I)
    denominator_mean = np.zeros(n_dims)
    denominator_cov = np.eye(n_dims)
    
    # P1 (numerator): N(mu, I) where ||mu||^2 = 2*KL
    # Distribute equally across all dimensions
    mu_norm_squared = 2.0 * target_kl
    mu_value = np.sqrt(mu_norm_squared / n_dims)
    numerator_mean = np.ones(n_dims) * mu_value
    numerator_cov = np.eye(n_dims)
    
    # Verify actual KL
    actual_kl = kl_between_two_gaussians(
        numerator_cov, denominator_cov,
        numerator_mean, denominator_mean
    )
    
    print(f"  Target KL: {target_kl:.2f}, Actual KL: {actual_kl:.4f}")
    print(f"  P1 mean: {numerator_mean[0]:.4f} * ones({n_dims})")
    print(f"  P0 mean: {denominator_mean[0]:.4f} * ones({n_dims})")
    
    return numerator_mean, numerator_cov, denominator_mean, denominator_cov, actual_kl


def create_tdre_config(n_samples, n_dims, target_kl):
    """
    Create TDRE configuration for training.
    
    Args:
        n_samples: Number of training samples
        n_dims: Dimensionality
        target_kl: Target KL divergence
        
    Returns:
        config object
    """
    # Scale hyperparameters based on sample size
    if n_samples <= 100:
        n_epochs = 500
        patience = 80
        lr = 1e-3
    elif n_samples <= 300:
        n_epochs = 400
        patience = 60
        lr = 5e-4
    else:
        n_epochs = 300
        patience = 50
        lr = 5e-4
    
    # Use fewer waymarks for simpler problems
    initial_waymark_indices = [0, 1, 2, 3, 4, 5]
    linear_combo_alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    config_dict = {
        'dataset_name': 'gaussians',
        'n_dims': n_dims,
        'data_seed': 42,
        'data_args': {
            'n_samples': n_samples,
            'n_dims': n_dims,
        },
        'frac': 1,
        'objective_nu': 1,
        'do_mutual_info_estimation': False,
        'data_dist_name': 'gaussian',
        'noise_dist_name': 'gaussian',
        'waymark_mechanism': 'linear_combinations',
        'shuffle_waymarks': False,
        'initial_waymark_indices': initial_waymark_indices,
        'linear_combo_alphas': linear_combo_alphas,
        'create_waymarks_in_zspace': False,
        'dimwise_mixing_strategy': 'fixed_single_order',
        'n_event_dims_to_mix': None,
        'waymark_mixing_increment': 1,
        'network_type': 'quadratic',
        'quadratic_constraint_type': 'symmetric_pos_diag',
        'quadratic_head_use_linear_term': True,
        'mlp_width': 128,
        'mlp_hidden_size': 128,
        'mlp_output_size': None,
        'mlp_n_blocks': 1,
        'n_mlp_layers': 2,
        'activation_name': 'relu',
        'loss_function': 'logistic',
        'optimizer': 'adam',
        'n_epochs': n_epochs,
        'n_batch': min(128, max(32, n_samples // 4)),
        'energy_lr': lr,
        'scale_param_lr_multiplier': 10.0,
        'energy_reg_coef': 0.0,
        'energy_restore_path': None,
        'patience': patience,
        'num_losses': len(initial_waymark_indices) - 1,
        'loss_decay_factor': 1.0,
        'save_dir': f'/tmp/tdre_mvgauss_kl{target_kl}_n{n_samples}/',
        'epoch_idx': -1,
        'use_residual_mlp': True,
        'use_cond_scale_shift': True,
        'shift_scale_per_channel': False,
        'use_instance_norm': False,
        'dropout_params': [False, 0.0, 0.0, 2.0],
        'max_spectral_norm_params': None,
        'just_track_spectral_norm': False,
        'label_smoothing_alpha': 0.0,
        'one_sided_smoothing': True,
        'save_every_x_epochs': None,
        'use_fc_layer': True,
        'final_pool_shape': (2, 2),
        'conv_kernel_shape': (3, 3),
        'use_global_sum_pooling': True,
        'use_attention': True,
        'head_type': 'quadratic',
        'use_single_head': False,
    }
    
    # Convert to AttrDict for compatibility with build_bridges
    class ConfigObject:
        def __init__(self, d):
            self.__dict__.update(d)
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
        
        def __getitem__(self, key):
            return self.__dict__[key]
        
        def __setitem__(self, key, value):
            self.__dict__[key] = value
        
        def __contains__(self, key):
            return key in self.__dict__
        
        def get(self, key, default=None):
            return self.__dict__.get(key, default)
        
        def keys(self):
            return self.__dict__.keys()
    
    return ConfigObject(config_dict)


def train_tdre_model(config, dataset):
    """
    Train a TDRE model.
    
    Args:
        config: Configuration object
        dataset: GAUSSIANS dataset object
        
    Returns:
        sess, graph, final_val_loss
    """
    print(f"    Building TDRE graph...")
    tf.reset_default_graph()
    tdre_graph = build_tdre_graph(config)
    
    # Create session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print(f"    Training TDRE for {config.n_epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    n_samples = config.data_args['n_samples']
    n_batches = max(1, n_samples // config.n_batch)
    
    for epoch in range(config.n_epochs):
        # Training
        for _ in range(n_batches):
            batch_size = min(config.n_batch, n_samples)
            idx = np.random.choice(n_samples, batch_size, replace=False)
            batch = dataset.trn.x[idx]
            
            feed_dict = {
                tdre_graph['data']: batch,
                tdre_graph['waymark_idxs']: config.initial_waymark_indices,
                tdre_graph['bridge_idxs']: config.initial_waymark_indices[:-1],
                tdre_graph['loss_weights']: np.ones(config.num_losses),
                tdre_graph['lr_var']: config.energy_lr,
            }
            
            sess.run(tdre_graph['tre_optim_op'], feed_dict=feed_dict)
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_batch_size = min(500, n_samples)
            val_batch = dataset.val.x[:val_batch_size]
            val_feed_dict = {
                tdre_graph['data']: val_batch,
                tdre_graph['waymark_idxs']: config.initial_waymark_indices,
                tdre_graph['bridge_idxs']: config.initial_waymark_indices[:-1],
                tdre_graph['loss_weights']: np.ones(config.num_losses),
            }
            
            val_loss_raw = sess.run(tdre_graph['val_loss'], feed_dict=val_feed_dict)
            val_loss = np.mean(val_loss_raw) if isinstance(val_loss_raw, np.ndarray) else val_loss_raw
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience // 10:
                print(f"      Early stopping at epoch {epoch}")
                break
    
    print(f"      Training complete (best val loss: {best_val_loss:.4f})")
    
    return sess, tdre_graph, best_val_loss


def evaluate_tdre(sess, graph, config, eval_data):
    """
    Evaluate TDRE model to get log ratio estimates.
    
    Args:
        sess: TensorFlow session
        graph: TDRE graph
        config: Config object
        eval_data: Evaluation samples from P1
        
    Returns:
        log_ratios: Array of log density ratio estimates
    """
    feed_dict = {
        graph['data']: eval_data,
        graph['waymark_idxs']: config.initial_waymark_indices,
        graph['bridge_idxs']: config.initial_waymark_indices[:-1],
    }
    
    # Get negative energies (log ratios for each bridge)
    neg_energies = sess.run(graph['neg_energies_of_data'], feed_dict=feed_dict)
    
    # Sum across all bridges to get total log ratio
    log_ratios = np.sum(neg_energies, axis=1)
    
    return log_ratios


def compute_relative_kl_error(estimated_kl, true_kl):
    """Compute relative error: |estimated - true| / |true|"""
    return np.abs(estimated_kl - true_kl) / np.abs(true_kl)


def run_experiment(target_kls=[10, 15, 20], 
                   sample_sizes=[50, 100, 300],
                   n_trials=10,
                   eval_size=10,
                   n_dims=10,
                   save_dir='results/multivariate_gaussian_experiment'):
    """
    Run the complete experiment.
    
    Args:
        target_kls: List of target KL divergences
        sample_sizes: List of training sample sizes
        n_trials: Number of independent trials
        eval_size: Number of evaluation samples
        n_dims: Dimensionality
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("MULTIVARIATE GAUSSIAN TDRE EXPERIMENT")
    print("="*80)
    print(f"Dimensions: {n_dims}")
    print(f"Target KLs: {target_kls}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Trials per config: {n_trials}")
    print(f"Evaluation size: {eval_size}")
    print("="*80)
    
    # Store results: results[kl][sample_size] = list of relative errors
    results = {kl: {n: [] for n in sample_sizes} for kl in target_kls}
    true_kls = {}
    
    # Run experiments for each KL divergence
    for target_kl in target_kls:
        print(f"\n{'='*80}")
        print(f"Target KL Divergence: {target_kl} nats")
        print(f"{'='*80}")
        
        # Create reference distribution
        num_mean, num_cov, den_mean, den_cov, actual_kl = create_gaussian_distributions_by_kl(
            target_kl, n_dims
        )
        true_kls[target_kl] = actual_kl
        
        # Run experiments for each sample size
        for n_samples in sample_sizes:
            print(f"\n{'-'*60}")
            print(f"Training Sample Size: {n_samples}")
            print(f"{'-'*60}")
            
            for trial in range(n_trials):
                print(f"\n  Trial {trial+1}/{n_trials}")
                
                # Create training dataset
                dataset = GAUSSIANS(
                    n_samples=n_samples,
                    n_dims=n_dims,
                    numerator_mean=num_mean,
                    numerator_cov=num_cov,
                    denominator_mean=den_mean,
                    denominator_cov=den_cov
                )
                
                # Add distribution parameters to dataset for evaluation
                dataset.data_args = {
                    'numerator_mean': num_mean,
                    'numerator_cov': num_cov,
                    'denominator_mean': den_mean,
                    'denominator_cov': den_cov
                }
                
                # Create config
                config = create_tdre_config(n_samples, n_dims, target_kl)
                config.data_args.update(dataset.data_args)
                
                # Train TDRE model
                sess, graph, val_loss = train_tdre_model(config, dataset)
                
                # Evaluate on fresh samples from P1
                eval_data = dataset.sample_data(eval_size)
                log_ratios = evaluate_tdre(sess, graph, config, eval_data)
                
                # Compute KL estimate
                estimated_kl = np.mean(log_ratios)
                relative_error = compute_relative_kl_error(estimated_kl, actual_kl)
                
                results[target_kl][n_samples].append(relative_error)
                
                print(f"    Estimated KL: {estimated_kl:.4f} (True: {actual_kl:.4f})")
                print(f"    Relative Error: {relative_error:.4f}")
                print(f"    Log ratios: mean={np.mean(log_ratios):.2f}, "
                      f"std={np.std(log_ratios):.2f}")
                
                # Close session
                sess.close()
                tf.reset_default_graph()
    
    # Compute summary statistics
    summary = {}
    for kl in target_kls:
        summary[kl] = {
            'sample_sizes': sample_sizes,
            'mean_errors': [np.mean(results[kl][n]) for n in sample_sizes],
            'std_errors': [np.std(results[kl][n]) for n in sample_sizes],
            'true_kl': true_kls[kl]
        }
    
    # Save results
    np.savez(
        os.path.join(save_dir, 'experiment_results.npz'),
        results=results,
        summary=summary,
        target_kls=target_kls,
        sample_sizes=sample_sizes,
        true_kls=true_kls,
        n_trials=n_trials,
        eval_size=eval_size,
        n_dims=n_dims
    )
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    return results, summary, true_kls


def plot_results(summary, save_path='results/multivariate_gaussian_experiment/sample_efficiency.pdf'):
    """
    Plot relative KL error vs training sample size for different KL values.
    
    Args:
        summary: Dictionary with summary statistics
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Colors and markers for different KL values
    colors = {10: '#E74C3C', 15: '#3498DB', 20: '#2ECC71'}
    markers = {10: 'o', 15: 's', 20: '^'}
    
    # Plot each KL divergence
    for kl in sorted(summary.keys()):
        sample_sizes = summary[kl]['sample_sizes']
        mean_errors = summary[kl]['mean_errors']
        std_errors = summary[kl]['std_errors']
        true_kl = summary[kl]['true_kl']
        
        # Plot mean with error bars
        ax.errorbar(
            sample_sizes, 
            mean_errors,
            yerr=std_errors,
            color=colors[kl],
            marker=markers[kl],
            markersize=10,
            linewidth=2.5,
            capsize=5,
            capthick=2,
            label=f'KL={true_kl:.1f} nats',
            alpha=0.9
        )
    
    ax.set_xlabel('Number of Training Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative KL Error', fontsize=14, fontweight='bold')
    ax.set_title('TDRE Sample Efficiency on 10D Multivariate Gaussians\n' + 
                 'Relative Error vs Training Sample Size',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Formatting
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_summary_table(summary):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    for kl in sorted(summary.keys()):
        print(f"\nKL = {summary[kl]['true_kl']:.2f} nats:")
        print(f"{'Sample Size':>15} | {'Mean Rel Error':>15} | {'Std Rel Error':>15}")
        print("-" * 52)
        
        for i, n in enumerate(summary[kl]['sample_sizes']):
            mean_err = summary[kl]['mean_errors'][i]
            std_err = summary[kl]['std_errors'][i]
            print(f"{n:>15} | {mean_err:>15.4f} | {std_err:>15.4f}")
    
    print("="*80)


def main():
    """Main experiment runner"""
    parser = ArgumentParser(description='Multivariate Gaussian TDRE Experiment')
    parser.add_argument('--target_kls', type=int, nargs='+', default=[10, 15, 20],
                       help='Target KL divergences (nats)')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[50, 100, 300],
                       help='Training sample sizes')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of trials per configuration')
    parser.add_argument('--eval_size', type=int, default=10,
                       help='Number of evaluation samples')
    parser.add_argument('--n_dims', type=int, default=10,
                       help='Dimensionality')
    parser.add_argument('--save_dir', type=str, 
                       default='results/multivariate_gaussian_experiment',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    # Run experiment
    results, summary, true_kls = run_experiment(
        target_kls=args.target_kls,
        sample_sizes=args.sample_sizes,
        n_trials=args.n_trials,
        eval_size=args.eval_size,
        n_dims=args.n_dims,
        save_dir=args.save_dir
    )
    
    # Print summary
    print_summary_table(summary)
    
    # Plot results
    plot_path = os.path.join(args.save_dir, 'sample_efficiency.pdf')
    plot_results(summary, save_path=plot_path)
    
    print(f"\n{'='*80}")
    print("ALL DONE!")
    print(f"Results and plots saved to: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


