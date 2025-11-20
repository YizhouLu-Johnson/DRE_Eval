"""
BDRE vs TDRE Comparison Script - FIXED VERSION

This script properly evaluates and compares BDRE and TDRE methods on 10D Gaussian
density ratio estimation tasks using THE SAME data distribution.

Key fixes:
1. Both methods use the same GAUSSIANS data distribution
2. TDRE models are actually loaded and evaluated
3. BDRE training is improved with proper validation
4. Data matches between training and evaluation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from argparse import ArgumentParser
from data_handlers.gaussians import GAUSSIANS
from train_bdre import BDREModel, build_bdre_graph
from utils.misc_utils import *
import json


def load_tdre_model(model_dir, config_path):
    """
    Load a trained TDRE model from disk.
    
    Args:
        model_dir: Directory containing the trained model
        config_path: Path to the config file
        
    Returns:
        session, graph, config
    """
    # Load config
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to object for build_bridges compatibility
    class ConfigObject:
        def __init__(self, d):
            self._dict = d
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
        
        def __contains__(self, key):
            return key in self._dict
        
        def __getitem__(self, key):
            return self._dict[key]
        
        def get(self, key, default=None):
            return self._dict.get(key, default)
    
    config = ConfigObject(config_dict)
    
    # Find checkpoint
    checkpoint_file = tf.train.latest_checkpoint(os.path.join(model_dir, 'model'))
    
    if checkpoint_file is None:
        raise ValueError(f"No checkpoint found in {model_dir}/model")
    
    # Import build_bridges to get the graph building function
    from build_bridges import build_graph
    
    # Build graph
    tf.reset_default_graph()
    graph = build_graph(config)
    
    # Create session and restore
    sess = tf.Session()
    
    # Only restore model variables (not optimizer variables like beta1_power)
    # Get all trainable variables from the graph
    model_vars = [v for v in tf.global_variables() if 'Adam' not in v.name and 'beta' not in v.name]
    saver = tf.train.Saver(var_list=model_vars)
    saver.restore(sess, checkpoint_file)
    
    print(f"  Loaded TDRE model from: {checkpoint_file}")
    
    return sess, graph, config


def evaluate_tdre_on_data(sess, graph, config, eval_data_p1, eval_data_p0):
    """
    Evaluate TDRE model to get log ratio estimates.
    
    Args:
        sess: TensorFlow session with loaded model
        graph: TDRE computational graph
        config: Config dictionary
        eval_data_p1: Samples from P1 (numerator)
        eval_data_p0: Samples from P0 (denominator)
        
    Returns:
        log_ratios: Estimated log density ratios
    """
    # For TDRE, the log ratio is the sum of negative energies across all bridges
    # neg_energies_of_data has shape (n_batch, n_ratios)
    
    n_samples = eval_data_p1.shape[0]
    n_dims = eval_data_p1.shape[1]
    
    # Create feed dict with the data placeholder
    feed_dict = {
        graph['data']: eval_data_p1,  # Evaluate on P1 samples
        graph['waymark_idxs']: config.initial_waymark_indices,
        graph['bridge_idxs']: config.initial_waymark_indices[:-1],
    }
    
    # Get negative energies of data
    # This gives us the log ratios for each bridge
    neg_energies = sess.run(graph['neg_energies_of_data'], feed_dict=feed_dict)
    
    # Sum across all bridges to get total log ratio
    # neg_energies has shape (n_samples, n_bridges)
    log_ratios = np.sum(neg_energies, axis=1)
    
    return log_ratios


def evaluate_bdre_on_data(sess, graph, eval_data_p1):
    """
    Evaluate BDRE model to get log ratio estimates.
    
    Args:
        sess: TensorFlow session with trained BDRE
        graph: BDRE computational graph  
        eval_data_p1: Samples from P1 to evaluate on
        
    Returns:
        log_ratios: Estimated log density ratios
    """
    feed_dict = {
        graph['x_p1']: eval_data_p1,
        graph['is_training']: False
    }
    log_ratios = sess.run(graph['log_ratio_p1'], feed_dict=feed_dict)
    return log_ratios


def compute_kl_estimate(log_ratios):
    """Compute KL divergence estimate from log ratios."""
    return np.mean(log_ratios)


def compute_relative_error(estimated_kl, true_kl):
    """Compute relative error in KL estimation."""
    return np.abs(estimated_kl - true_kl) / np.abs(true_kl)


def run_comparison_experiment(
    sample_sizes=[50, 100, 400, 800, 1600, 3200],
    eval_sample_sizes=[10, 100],
    n_trials=30,
    n_dims=10,
    true_mi=5.0,
    tdre_model_dir=None,
    save_dir='results/bdre_tdre_comparison_fixed'
):
    """
    Run the complete comparison experiment.
    
    Args:
        sample_sizes: List of training sample sizes to test
        eval_sample_sizes: List of evaluation sample sizes (M values)
        n_trials: Number of independent trials per configuration
        n_dims: Dimensionality of the data
        true_mi: True mutual information (controls KL divergence)
        tdre_model_dir: Directory containing trained TDRE model
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create reference dataset to get true KL
    reference_dataset = GAUSSIANS(n_samples=10000, n_dims=n_dims, true_mutual_info=true_mi)
    true_kl = kl_between_two_gaussians(reference_dataset.cov_matrix, reference_dataset.denom_cov_matrix)
    
    print(f"True KL divergence: {true_kl:.4f}")
    print(f"Dimensionality: {n_dims}")
    print(f"True mutual info: {true_mi}")
    
    # Load TDRE model if available
    tdre_sess = None
    tdre_graph = None
    if tdre_model_dir is not None and os.path.exists(tdre_model_dir):
        print(f"\nLoading TDRE model from: {tdre_model_dir}")
        tdre_sess, tdre_graph, tdre_config = load_tdre_model(tdre_model_dir, None)
        print("TDRE model loaded successfully!")
    else:
        print("\nWarning: No TDRE model specified or found. Will only evaluate BDRE.")
    
    # Initialize results storage
    results = {
        'bdre': {M: {n: [] for n in sample_sizes} for M in eval_sample_sizes},
        'tdre': {M: {n: [] for n in sample_sizes} for M in eval_sample_sizes},
        'true_kl': true_kl,
        'sample_sizes': sample_sizes,
        'eval_sample_sizes': eval_sample_sizes
    }
    
    # Run experiments for each sample size
    for n_samples in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Training Sample Size: n = {n_samples}")
        print(f"{'='*80}")
        
        for trial in range(n_trials):
            print(f"\nTrial {trial+1}/{n_trials}")
            
            # Create training dataset - USE SAME DISTRIBUTION AS TDRE
            trial_dataset = GAUSSIANS(
                n_samples=n_samples,
                n_dims=n_dims,
                true_mutual_info=true_mi
            )
            
            # ==================== Train BDRE ====================
            print("  Training BDRE...")
            tf.reset_default_graph()
            
            bdre_config = {
                'input_dim': n_dims,
                'hidden_dims': [128, 128, 128],  # 3 layers
                'n_train_samples': n_samples,
                'batch_size': min(64, max(16, n_samples // 4)),
                'n_epochs': 300,  # More epochs
                'lr': 1e-3,
                'patience': 30,  # More patience
                'activation': 'relu',
            }
            
            # Build BDRE graph
            bdre_graph, bdre_model = build_bdre_graph(bdre_config)
            bdre_sess = tf.Session()
            bdre_sess.run(tf.global_variables_initializer())
            
            # Training loop with validation
            n_batches = max(1, n_samples // bdre_config['batch_size'])
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(bdre_config['n_epochs']):
                # Training
                epoch_losses = []
                for _ in range(n_batches):
                    # Sample batch from numerator (P1) - using training data
                    batch_size = min(bdre_config['batch_size'], n_samples)
                    idx = np.random.choice(n_samples, batch_size, replace=False)
                    x_p1_batch = trial_dataset.trn.x[idx]
                    
                    # Sample from denominator (P0) - generate fresh samples
                    x_p0_batch = trial_dataset.sample_denominator(batch_size)
                    
                    feed_dict = {
                        bdre_graph['x_p0']: x_p0_batch,
                        bdre_graph['x_p1']: x_p1_batch,
                        bdre_graph['is_training']: True
                    }
                    
                    _, loss = bdre_sess.run(
                        [bdre_graph['train_op'], bdre_graph['loss']], 
                        feed_dict=feed_dict
                    )
                    epoch_losses.append(loss)
                
                # Validation every 10 epochs
                if epoch % 10 == 0:
                    val_batch_size = min(500, n_samples)
                    val_feed_dict = {
                        bdre_graph['x_p0']: trial_dataset.sample_denominator(val_batch_size),
                        bdre_graph['x_p1']: trial_dataset.val.x[:val_batch_size],
                        bdre_graph['is_training']: False
                    }
                    val_loss = bdre_sess.run(bdre_graph['loss'], feed_dict=val_feed_dict)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= bdre_config['patience'] // 10:
                        # print(f"    Early stopping at epoch {epoch}")
                        break
            
            # ==================== Evaluate Both Methods ====================
            for M in eval_sample_sizes:
                # Get evaluation samples from P1
                eval_samples_p1 = trial_dataset.sample_data(M)
                eval_samples_p0 = trial_dataset.sample_denominator(M)
                
                # Evaluate BDRE
                bdre_log_ratios = evaluate_bdre_on_data(bdre_sess, bdre_graph, eval_samples_p1)
                bdre_kl = compute_kl_estimate(bdre_log_ratios)
                bdre_error = compute_relative_error(bdre_kl, true_kl)
                results['bdre'][M][n_samples].append(bdre_error)
                
                print(f"    BDRE (M={M}): KL={bdre_kl:.4f}, Rel Error={bdre_error:.4f}")
                
                # Evaluate TDRE if available
                if tdre_sess is not None:
                    tdre_log_ratios = evaluate_tdre_on_data(
                        tdre_sess, tdre_graph, tdre_config,
                        eval_samples_p1, eval_samples_p0
                    )
                    tdre_kl = compute_kl_estimate(tdre_log_ratios)
                    tdre_error = compute_relative_error(tdre_kl, true_kl)
                    results['tdre'][M][n_samples].append(tdre_error)
                    
                    print(f"    TDRE (M={M}): KL={tdre_kl:.4f}, Rel Error={tdre_error:.4f}")
            
            # Close BDRE session
            bdre_sess.close()
    
    # Compute summary statistics
    summary = {
        'bdre': {},
        'tdre': {},
        'true_kl': true_kl
    }
    
    for method in ['bdre', 'tdre']:
        summary[method] = {}
        for M in eval_sample_sizes:
            summary[method][M] = {
                'mean': [],
                'std': [],
                'sample_sizes': sample_sizes
            }
            for n in sample_sizes:
                errors = results[method][M][n]
                if len(errors) > 0:
                    summary[method][M]['mean'].append(np.mean(errors))
                    summary[method][M]['std'].append(np.std(errors))
                else:
                    summary[method][M]['mean'].append(np.nan)
                    summary[method][M]['std'].append(np.nan)
    
    # Save results
    np.savez(
        os.path.join(save_dir, 'comparison_results.npz'),
        **results,
        summary=summary
    )
    
    # Close TDRE session if it was loaded
    if tdre_sess is not None:
        tdre_sess.close()
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    return results, summary


def plot_comparison(summary, save_path='results/bdre_tdre_comparison_fixed/comparison_plot.pdf'):
    """
    Plot relative KL error comparison between BDRE and TDRE.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different M values
    colors = {10: 'red', 100: 'blue'}
    
    # Plot BDRE results
    for M in summary['bdre'].keys():
        sample_sizes = summary['bdre'][M]['sample_sizes']
        mean_errors = summary['bdre'][M]['mean']
        std_errors = summary['bdre'][M]['std']
        
        # Filter out NaNs
        valid_idx = ~np.isnan(mean_errors)
        if np.any(valid_idx):
            ax.plot(np.array(sample_sizes)[valid_idx], np.array(mean_errors)[valid_idx], 
                    color=colors[M], linestyle='-', linewidth=2,
                    marker='o', markersize=6,
                    label=f'BDRE (M={M})')
            # ax.fill_between(np.array(sample_sizes)[valid_idx], 
            #                 np.array(mean_errors)[valid_idx] - np.array(std_errors)[valid_idx],
            #                 np.array(mean_errors)[valid_idx] + np.array(std_errors)[valid_idx],
            #                 color=colors[M], alpha=0.2)
    
    # Plot TDRE results if available
    if 'tdre' in summary and len(summary['tdre']) > 0:
        has_tdre = False
        for M in summary['tdre'].keys():
            sample_sizes = summary['tdre'][M]['sample_sizes']
            mean_errors = summary['tdre'][M]['mean']
            std_errors = summary['tdre'][M]['std']
            
            # Filter out NaNs
            valid_idx = ~np.isnan(mean_errors)
            if np.any(valid_idx):
                has_tdre = True
                ax.plot(np.array(sample_sizes)[valid_idx], np.array(mean_errors)[valid_idx],
                        color=colors[M], linestyle='--', linewidth=2,
                        marker='s', markersize=6,
                        label=f'TDRE (M={M})')
                # ax.fill_between(np.array(sample_sizes)[valid_idx],
                #                 np.array(mean_errors)[valid_idx] - np.array(std_errors)[valid_idx],
                #                 np.array(mean_errors)[valid_idx] + np.array(std_errors)[valid_idx],
                #                 color=colors[M], alpha=0.2)
        
        if not has_tdre:
            ax.text(0.5, 0.95, 'TDRE results not available', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Number of Training Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Relative KL Error', fontsize=12)
    ax.set_title('BDRE vs TDRE: Sample Efficiency Comparison\n10D Gaussian Density Ratio Estimation', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()


def main():
    """Main comparison script"""
    parser = ArgumentParser(description='Compare BDRE and TDRE methods (FIXED VERSION)')
    parser.add_argument('--sample_sizes', type=int, nargs='+', 
                       default=[50, 100, 400, 800, 1600, 3200],
                       help='Training sample sizes to test')
    parser.add_argument('--eval_sample_sizes', type=int, nargs='+',
                       default=[10, 100],
                       help='Evaluation sample sizes (M values)')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of trials per configuration')
    parser.add_argument('--n_dims', type=int, default=10,
                       help='Data dimensionality')
    parser.add_argument('--true_mi', type=float, default=5.0,
                       help='True mutual information (controls KL)')
    parser.add_argument('--tdre_model_dir', type=str, default=None,
                       help='Directory containing trained TDRE model')
    parser.add_argument('--save_dir', type=str, 
                       default='results/bdre_tdre_comparison_fixed',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run experiment
    results, summary = run_comparison_experiment(
        sample_sizes=args.sample_sizes,
        eval_sample_sizes=args.eval_sample_sizes,
        n_trials=args.n_trials,
        n_dims=args.n_dims,
        true_mi=args.true_mi,
        tdre_model_dir=args.tdre_model_dir,
        save_dir=args.save_dir
    )
    
    # Plot results
    plot_path = os.path.join(args.save_dir, 'comparison_plot.pdf')
    plot_comparison(summary, save_path=plot_path)
    
    print("\nComparison complete!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

