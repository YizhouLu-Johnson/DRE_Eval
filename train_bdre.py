"""
Binary Density Ratio Estimation (BDRE) Training Script

This script trains a binary classifier to estimate density ratios between two distributions.
The classifier learns to distinguish samples from P1 (numerator) vs P0 (denominator).
At optimum, the classifier output approximates log(P1(x)/P0(x)).
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from argparse import ArgumentParser
from data_handlers.two_gaussians import TwoGaussians
from utils.misc_utils import *
from utils.tf_utils import *


class BDREModel:
    """
    Binary classification-based density ratio estimator.
    Uses a neural network to classify samples from P1 vs P0.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 128], activation='relu', reg_coef=0.0, dropout_rate=0.0):
        """
        Initialize BDRE model.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.reg_coef = float(reg_coef)
        self.dropout_rate = float(dropout_rate)
        
    def build_network(self, x, is_training=True):
        """
        Build neural network for binary classification.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            is_training: Whether in training mode
            
        Returns:
            logits: Raw classifier output (log density ratio estimate)
        """
        with tf.compat.v1.variable_scope("bdre_classifier", reuse=tf.compat.v1.AUTO_REUSE):
            h = x
            
            # Hidden layers
            for i, hidden_dim in enumerate(self.hidden_dims):
                kernel_reg = tf.keras.regularizers.l2(self.reg_coef) if self.reg_coef > 0.0 else None
                h = tf.keras.layers.Dense(
                    hidden_dim, 
                    activation=None,
                    kernel_regularizer=kernel_reg,
                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                    name=f'hidden_{i}'
                )(h)
                
                # Apply activation
                if self.activation == 'relu':
                    h = tf.nn.relu(h)
                elif self.activation == 'tanh':
                    h = tf.nn.tanh(h)
                elif self.activation == 'elu':
                    h = tf.nn.elu(h)
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
                
                # Optional: Add dropout for regularization
                if self.dropout_rate > 0.0:
                    h = tf.keras.layers.Dropout(self.dropout_rate)(h, training=is_training)
            
            # Output layer (no activation - raw logits)
            kernel_reg = tf.keras.regularizers.l2(self.reg_coef) if self.reg_coef > 0.0 else None
            logits = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_regularizer=kernel_reg,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                name='output'
            )(h)
            
            logits = tf.squeeze(logits, axis=-1)  # [batch_size]
            
        return logits


def build_bdre_graph(config):
    """
    Build TensorFlow graph for BDRE training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing graph components
    """
    graph = {}
    
    # Placeholders
    graph['x_p1'] = tf.compat.v1.placeholder(tf.float32, [None, config['input_dim']], name='x_p1')
    graph['x_p0'] = tf.compat.v1.placeholder(tf.float32, [None, config['input_dim']], name='x_p0')
    graph['learning_rate'] = tf.compat.v1.placeholder_with_default(config['lr'], (), name='lr')
    graph['is_training'] = tf.compat.v1.placeholder_with_default(True, (), name='is_training')
    
    # Build model
    model = BDREModel(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        activation=config.get('activation', 'relu'),
        reg_coef=config.get('reg_coef', 0.0),
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    # Get log density ratio estimates
    # For P1 samples: log r(x) = log P1(x) / P0(x)
    # For P0 samples: log r(x) = log P1(x) / P0(x)
    log_ratio_p1 = model.build_network(graph['x_p1'], is_training=graph['is_training'])
    log_ratio_p0 = model.build_network(graph['x_p0'], is_training=graph['is_training'])
    
    graph['log_ratio_p1'] = log_ratio_p1
    graph['log_ratio_p0'] = log_ratio_p0
    
    # Binary cross-entropy loss
    # For P1 samples (label=1): -log(sigmoid(log_ratio))
    # For P0 samples (label=0): -log(1 - sigmoid(log_ratio)) = -log(sigmoid(-log_ratio))
    loss_p1 = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(log_ratio_p1),
        logits=log_ratio_p1
    )
    loss_p0 = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(log_ratio_p0),
        logits=log_ratio_p0
    )
    
    graph['loss'] = tf.reduce_mean(loss_p1) + tf.reduce_mean(loss_p0)
    
    # Accuracy metrics
    pred_p1 = tf.cast(log_ratio_p1 > 0, tf.float32)
    pred_p0 = tf.cast(log_ratio_p0 < 0, tf.float32)
    graph['acc_p1'] = tf.reduce_mean(pred_p1)
    graph['acc_p0'] = tf.reduce_mean(pred_p0)
    graph['acc'] = (graph['acc_p1'] + graph['acc_p0']) / 2.0
    
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(graph['learning_rate'])
    graph['train_op'] = optimizer.minimize(graph['loss'])

    # Clip gradients by global norm (e.g., 5.0). You can tune this value if needed.
    # trainable_vars = tf.compat.v1.trainable_variables()
    # grads = tf.gradients(graph['loss'], trainable_vars)
    # clipped_grads, _ = tf.clip_by_global_norm(grads, 1)
    # graph['train_op'] = optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
    
    
    return graph, model


def train_bdre(config, dataset, save_dir):
    """
    Train BDRE model.
    
    Args:
        config: Configuration dictionary
        dataset: TwoGaussians dataset object
        save_dir: Directory to save model
        
    Returns:
        Trained session and graph
    """
    print(f"\n{'='*60}")
    print(f"Training BDRE Model")
    print(f"{'='*60}")
    print(f"Input dim: {config['input_dim']}")
    print(f"Hidden dims: {config['hidden_dims']}")
    print(f"Training samples: {config['n_train_samples']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['n_epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")
    
    # Build graph
    tf.compat.v1.reset_default_graph()
    graph, model = build_bdre_graph(config)
    
    # Create session
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Saver
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    
    # Training loop
    n_batches = max(1, config['n_train_samples'] // config['batch_size'])
    best_val_loss = np.inf
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['n_epochs']):
        epoch_losses = []
        epoch_accs = []
        
        # Shuffle training data
        idx_p0 = np.random.permutation(dataset.trn.N_p0)
        idx_p1 = np.random.permutation(dataset.trn.N_p1)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config['batch_size']
            end_idx = start_idx + config['batch_size']
            
            # Get batch
            batch_p0 = dataset.trn.x_p0[idx_p0[start_idx:end_idx]]
            batch_p1 = dataset.trn.x_p1[idx_p1[start_idx:end_idx]]
            
            # Train step
            feed_dict = {
                graph['x_p0']: batch_p0,
                graph['x_p1']: batch_p1,
                graph['is_training']: True
            }
            
            _, loss_val, acc_val = sess.run(
                [graph['train_op'], graph['loss'], graph['acc']],
                feed_dict=feed_dict
            )
            
            epoch_losses.append(loss_val)
            epoch_accs.append(acc_val)
        
        # Compute validation loss
        val_feed_dict = {
            graph['x_p0']: dataset.val.x_p0[:config['batch_size']*4],
            graph['x_p1']: dataset.val.x_p1[:config['batch_size']*4],
            graph['is_training']: False
        }
        val_loss = sess.run(graph['loss'], feed_dict=val_feed_dict)
        
        train_losses.append(np.mean(epoch_losses))
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{config['n_epochs']} | "
                  f"Train Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {np.mean(epoch_accs):.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            saver.save(sess, os.path.join(save_dir, 'bdre_model.ckpt'))
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    saver.save(sess, os.path.join(save_dir, 'bdre_model_final.ckpt'))
    
    # Save training history
    np.savez(
        os.path.join(save_dir, 'training_history.npz'),
        train_losses=train_losses,
        val_losses=val_losses
    )
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")
    
    return sess, graph, model


def main():
    """Main training function"""
    parser = ArgumentParser()
    parser.add_argument('--n_train_samples', type=int, default=1000, 
                       help='Number of training samples per distribution')
    parser.add_argument('--n_dims', type=int, default=10,
                       help='Data dimensionality')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--mean_shift', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='saved_models/bdre_comparison/bdre')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    
    # Create dataset
    print("Generating dataset...")
    dataset = TwoGaussians(
        n_samples=max(args.n_train_samples, 5000),  # Ensure enough for train/val/test
        n_dims=args.n_dims,
        mean_shift=args.mean_shift,
        seed=args.seed
    )
    
    print(f"True KL(P1||P0): {dataset.true_kl:.4f}")
    
    # Prepare config
    config = {
        'input_dim': args.n_dims,
        'hidden_dims': args.hidden_dims,
        'n_train_samples': args.n_train_samples,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'patience': args.patience,
        'activation': args.activation,
    }
    
    # Train model
    sess, graph, model = train_bdre(config, dataset, args.save_dir)
    
    # Save dataset info
    dataset_info = {
        'n_dims': args.n_dims,
        'mean_shift': args.mean_shift,
        'true_kl': dataset.true_kl,
        'mu0': dataset.mu0,
        'mu1': dataset.mu1,
        'cov0': dataset.cov0,
        'cov1': dataset.cov1,
    }
    np.savez(os.path.join(args.save_dir, 'dataset_info.npz'), **dataset_info)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
