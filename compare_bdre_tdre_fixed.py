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
from build_bridges import build_graph as build_tdre_graph
import json
import logging


def train_tdre_model(n_samples, n_dims, use_distinct_distributions, reference_dataset, trial_dataset, base_config):
    """
    Train a TDRE model from scratch for the given sample size.
    
    Args:
        n_samples: Number of training samples
        n_dims: Dimensionality
        use_distinct_distributions: Whether using distinct distributions
        reference_dataset: Reference dataset for distribution parameters
        trial_dataset: Trial-specific dataset
        base_config: Base configuration dictionary for TDRE
        
    Returns:
        session, graph, config
    """
    # Create TDRE configuration
    from utils.misc_utils import AttrDict
    
    # Simple AttrDict that allows both dict and attribute access
    class TDREConfig:
        def __init__(self, d):
            self.__dict__.update(d)
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, TDREConfig(value))
        
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
    
    # Scale hyperparameters based on sample size (similar to BDRE)
    if n_samples <= 100:
        n_epochs = 300
        patience = 50
    elif n_samples <= 800:
        n_epochs = 250
        patience = 40
    else:
        n_epochs = 300
        patience = 50
    
    # Create config for TDRE
    initial_waymark_indices = base_config.get('initial_waymark_indices', [0, 1, 2, 3, 4, 5])
    linear_combo_alphas = base_config.get('linear_combo_alphas', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
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
        'mlp_hidden_size': 128,  # Alias
        'mlp_output_size': None,
        'mlp_n_blocks': 1,
        'n_mlp_layers': 3,
        'activation_name': 'relu',
        'loss_function': 'logistic',
        'optimizer': 'adam',
        'n_epochs': n_epochs,
        'n_batch': min(128, max(32, n_samples // 4)),
        'energy_lr': 5e-4,
        'scale_param_lr_multiplier': 10.0,
        'energy_reg_coef': 0.0,
        'energy_restore_path': None,
        'patience': patience,
        'num_losses': len(initial_waymark_indices) - 1,
        'loss_decay_factor': 1.0,
        'save_dir': '/tmp/tdre_temp/',
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
    
    # Add distribution parameters
    if use_distinct_distributions:
        config_dict['data_args'].update({
            'numerator_mean': reference_dataset.means.tolist(),
            'numerator_cov': reference_dataset.cov_matrix.tolist(),
            'denominator_mean': reference_dataset.denom_means.tolist(),
            'denominator_cov': reference_dataset.denom_cov_matrix.tolist(),
        })
    
    config = TDREConfig(config_dict)
    
    # Build TDRE graph
    tf.reset_default_graph()
    tdre_graph = build_tdre_graph(config)
    
    # Create session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Simple training loop (simplified version of build_bridges.train)
    print(f"    Training TDRE (n={n_samples})...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    n_batches = max(1, n_samples // config.n_batch)
    
    for epoch in range(config.n_epochs):
        # Training
        for _ in range(n_batches):
            batch_size = min(config.n_batch, n_samples)
            idx = np.random.choice(n_samples, batch_size, replace=False)
            batch = trial_dataset.trn.x[idx]
            
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
            val_batch = trial_dataset.val.x[:val_batch_size]
            val_feed_dict = {
                tdre_graph['data']: val_batch,
                tdre_graph['waymark_idxs']: config.initial_waymark_indices,
                tdre_graph['bridge_idxs']: config.initial_waymark_indices[:-1],
                tdre_graph['loss_weights']: np.ones(config.num_losses),
            }
            
            val_loss_raw = sess.run(tdre_graph['val_loss'], feed_dict=val_feed_dict)
            # val_loss might be an array (one per bridge), take mean
            val_loss = np.mean(val_loss_raw) if isinstance(val_loss_raw, np.ndarray) else val_loss_raw
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience // 10:
                print(f"      Early stopping at epoch {epoch}")
                break
    
    print(f"      Trained for {epoch+1} epochs")
    
    return sess, tdre_graph, config


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
    eval_sample_sizes=[10, 100, 1000],
    n_trials=30,
    n_dims=10,
    true_mi=5.0,
    tdre_model_dir=None,
    save_dir='results/updated_result',
    use_distinct_distributions=True
):
    """
    Run the complete comparison experiment.
    
    **IMPORTANT:** Both BDRE and TDRE are now trained fresh for EACH sample size!
    This ensures a fair comparison of sample efficiency.
    
    Args:
        sample_sizes: List of training sample sizes to test
        eval_sample_sizes: List of evaluation sample sizes (M values)
        n_trials: Number of independent trials per configuration
        n_dims: Dimensionality of the data
        true_mi: True mutual information (controls KL divergence) - DEPRECATED when use_distinct_distributions=True
        tdre_model_dir: OPTIONAL directory to load TDRE waymark configuration from (not for pre-trained model!)
        save_dir: Directory to save results
        use_distinct_distributions: If True, use distinct Gaussians with different means/variances (KL >= 10 nats)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create reference dataset to get true KL
    if use_distinct_distributions:
        # Choose distinct Gaussians so KL(P1||P0) is roughly within 10-15 nats.
        # For diagonal covariances equal to I, KL reduces to 0.5 * d * a^2 for means a*ones.
        a = 1.45
        numerator_mean = np.ones(n_dims) * float(a)
        numerator_cov = np.eye(n_dims)
        denominator_mean = np.zeros(n_dims)
        denominator_cov = np.eye(n_dims)

        reference_dataset = GAUSSIANS(
            n_samples=10000,
            n_dims=n_dims,
            numerator_mean=numerator_mean,
            numerator_cov=numerator_cov,
            denominator_mean=denominator_mean,
            denominator_cov=denominator_cov
        )
    else:
        # Old way: use correlation-based MI (distributions too similar!)
        reference_dataset = GAUSSIANS(n_samples=10000, n_dims=n_dims, true_mutual_info=true_mi)
    
    # Compute analytical KL divergence
    true_kl = kl_between_two_gaussians(
        reference_dataset.cov_matrix,
        reference_dataset.denom_cov_matrix,
        reference_dataset.means,
        reference_dataset.denom_means,
    )

    print(f"=" * 80)
    print(f"EXPERIMENT SETUP")
    print(f"=" * 80)
    print(f"Dimensionality: {n_dims}")
    if use_distinct_distributions:
        print(f"Using DISTINCT Gaussian distributions:")
        print(f"  P (numerator): mean={numerator_mean[0]:.2f}*ones, cov=I")
        print(f"  Q (denominator): mean={denominator_mean[0]:.2f}*ones, cov=I")
    else:
        print(f"Using correlation-based MI: {true_mi}")
    print(f"Analytical KL(P||Q): {true_kl:.4f} nats")
    print(f"Analytical KL(P||Q): {true_kl:.4f} nats")
    print(f"=" * 80)
    
    # Load base TDRE config if provided (for waymark settings)
    base_tdre_config = {}
    if tdre_model_dir is not None and os.path.exists(tdre_model_dir):
        config_path = os.path.join(tdre_model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Extract waymark configuration
                if 'data' in loaded_config:
                    base_tdre_config['initial_waymark_indices'] = loaded_config['data'].get('initial_waymark_indices', [0, 1, 2, 3, 4, 5])
                    base_tdre_config['linear_combo_alphas'] = loaded_config['data'].get('linear_combo_alphas', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                print(f"\nLoaded TDRE waymark configuration from: {tdre_model_dir}")
                print(f"  Waymarks: {base_tdre_config.get('initial_waymark_indices', [0, 1, 2, 3, 4, 5])}")
        else:
            print("\nWarning: No TDRE config found. Will use default waymark settings.")
    else:
        print("\nNo TDRE model directory specified. Using default TDRE waymark settings.")
        base_tdre_config = {
            'initial_waymark_indices': [0, 1, 2, 3, 4, 5],
            'linear_combo_alphas': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }
        print(f"  Waymarks: {base_tdre_config['initial_waymark_indices']}")
    
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
            
            # Create training dataset - USE SAME DISTRIBUTION AS REFERENCE
            if use_distinct_distributions:
                trial_dataset = GAUSSIANS(
                    n_samples=n_samples,
                    n_dims=n_dims,
                    numerator_mean=reference_dataset.means,
                    numerator_cov=reference_dataset.cov_matrix,
                    denominator_mean=reference_dataset.denom_means,
                    denominator_cov=reference_dataset.denom_cov_matrix
                )
            else:
                trial_dataset = GAUSSIANS(
                    n_samples=n_samples,
                    n_dims=n_dims,
                    true_mutual_info=true_mi
                )
            
            # ==================== Train BDRE ====================
            print(f"  Training BDRE (n={n_samples})...")
            tf.reset_default_graph()
            
            # Scale hyperparameters based on sample size for better convergence
            # Smaller datasets need more epochs per sample, larger datasets need more patience
            if n_samples <= 100:
                n_epochs = 300
                patience_checks = 10
                lr = 1e-3
            elif n_samples <= 800:
                n_epochs = 200
                patience_checks = 10
                lr = 5e-4
            else:  # Large datasets
                n_epochs = 200
                patience_checks = 10
                lr = 2e-4  # Lower LR for stability with more data
            
            # regularisation to prevent overfitting on large models / large samples
            if n_samples <= 100:
                hidden_dims = [128, 128]
                reg_coef = 1e-4
                dropout_rate = 0.2
            elif n_samples <= 800:
                hidden_dims = [128, 128]
                reg_coef = 5e-4
                dropout_rate = 0.1
            else:
                hidden_dims = [128, 128]
                reg_coef = 1e-3
                dropout_rate = 0.05

            bdre_config = {
                'input_dim': n_dims,
                'hidden_dims': hidden_dims,
                'n_train_samples': n_samples,
                'batch_size': min(128, max(32, n_samples // 4)),
                'n_epochs': n_epochs,
                'lr': lr,
                'patience_checks': patience_checks,  # Number of validation checks without improvement
                'activation': 'relu',
                'reg_coef': reg_coef,
                'dropout_rate': dropout_rate,
            }
            
            # Build BDRE graph
            bdre_graph, bdre_model = build_bdre_graph(bdre_config)
            bdre_sess = tf.Session()
            bdre_sess.run(tf.global_variables_initializer())
            
            # Training loop with validation
            n_batches = max(1, n_samples // bdre_config['batch_size'])
            best_val_loss = float('inf')
            patience_counter = 0
            val_check_freq = 10  # Check validation every 10 epochs
            
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
                if epoch % val_check_freq == 0:
                    # Use more validation samples for better estimates
                    val_batch_size = min(1000, max(100, n_samples))
                    val_feed_dict = {
                        bdre_graph['x_p0']: trial_dataset.sample_denominator(val_batch_size),
                        bdre_graph['x_p1']: trial_dataset.val.x[:val_batch_size],
                        bdre_graph['is_training']: False
                    }
                    val_loss = bdre_sess.run(bdre_graph['loss'], feed_dict=val_feed_dict)
                    
                    # Early stopping with improved patience
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    # Stop if no improvement for 'patience_checks' validation checks
                    if patience_counter >= bdre_config['patience_checks']:
                        print(f"    Early stopping at epoch {epoch} (no improvement for {patience_counter * val_check_freq} epochs)")
                        break
            
            # Print final training info
            if epoch < bdre_config['n_epochs'] - 1:
                print(f"    Trained for {epoch+1} epochs (early stopped)")
            else:
                print(f"    Trained for {epoch+1} epochs (full training)")
            
            # ==================== Train TDRE ====================
            # Train TDRE from scratch with the same sample size
            tdre_sess, tdre_graph, tdre_config = train_tdre_model(
                n_samples=n_samples,
                n_dims=n_dims,
                use_distinct_distributions=use_distinct_distributions,
                reference_dataset=reference_dataset,
                trial_dataset=trial_dataset,
                base_config=base_tdre_config
            )
            
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
                
                # Show log ratio statistics to diagnose issues
                print(f"    BDRE (M={M}): KL={bdre_kl:.4f} (true={true_kl:.4f}), " + 
                      f"Rel Err={bdre_error:.4f}, log_ratio: mean={np.mean(bdre_log_ratios):.2f}±{np.std(bdre_log_ratios):.2f}")
                
                # Evaluate TDRE (now always available since we train it)
                tdre_log_ratios = evaluate_tdre_on_data(
                    tdre_sess, tdre_graph, tdre_config,
                    eval_samples_p1, eval_samples_p0
                )
                tdre_kl = compute_kl_estimate(tdre_log_ratios)
                tdre_error = compute_relative_error(tdre_kl, true_kl)
                results['tdre'][M][n_samples].append(tdre_error)
                
                # Show log ratio statistics to diagnose issues
                print(f"    TDRE (M={M}): KL={tdre_kl:.4f} (true={true_kl:.4f}), " + 
                      f"Rel Err={tdre_error:.4f}, log_ratio: mean={np.mean(tdre_log_ratios):.2f}±{np.std(tdre_log_ratios):.2f}")
            
            # Close sessions
            bdre_sess.close()
            tdre_sess.close()
    
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
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {save_dir}")
    print(f"Note: Both BDRE and TDRE were trained fresh for each sample size")
    print(f"{'='*80}\n")
    
    return results, summary


def plot_comparison(summary, save_path='results/bdre_tdre_comparison_fixed/comparison_plot.pdf'):
    """
    Plot relative KL error comparison between BDRE and TDRE.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different M values
    colors = {10: 'red', 100: 'blue', 1000: 'green'}
    
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
    
    # Get KL divergence from summary if available
    kl_str = f" (KL={summary['true_kl']:.2f} nats)" if 'true_kl' in summary else ""
    ax.set_title(f'BDRE vs TDRE: Sample Efficiency Comparison\n10D Gaussian Density Ratio Estimation{kl_str}', 
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
                       default=[10, 100, 1000],
                       help='Evaluation sample sizes (M values)')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of trials per configuration')
    parser.add_argument('--n_dims', type=int, default=10,
                       help='Data dimensionality')
    parser.add_argument('--true_mi', type=float, default=5.0,
                       help='True mutual information (controls KL)')
    parser.add_argument('--tdre_model_dir', type=str, default=None,
                       help='OPTIONAL: Directory with TDRE config to load waymark settings (TDRE trains fresh for each sample size!)')
    parser.add_argument('--save_dir', type=str, 
                       default='results/bdre_tdre_comparison_fixed',
                       help='Directory to save results')
    parser.add_argument('--use_distinct_distributions', action='store_true', default=True,
                       help='Use distinct Gaussian distributions with different means/variances (KL >= 10 nats)')
    parser.add_argument('--no_distinct_distributions', action='store_false', dest='use_distinct_distributions',
                       help='Use old correlation-based distributions (not recommended)')
    
    args = parser.parse_args()
    
    # Run experiment
    results, summary = run_comparison_experiment(
        sample_sizes=args.sample_sizes,
        eval_sample_sizes=args.eval_sample_sizes,
        n_trials=args.n_trials,
        n_dims=args.n_dims,
        true_mi=args.true_mi,
        tdre_model_dir=args.tdre_model_dir,
        save_dir=args.save_dir,
        use_distinct_distributions=args.use_distinct_distributions
    )
    
    # Plot results
    plot_path = os.path.join(args.save_dir, 'comparison_plot.pdf')
    plot_comparison(summary, save_path=plot_path)
    
    print("\nComparison complete!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

