"""
Train TSM (Time Score Matching) models for the gaussians_10d experiments.

This implements the DRE-infinity approach from:
"Density Ratio Estimation via Infinitesimal Classification" (Choi et al., AISTATS 2022)

The key idea: Instead of learning a binary classifier, TSM learns the time-score
∂log p(x_t)/∂t along a stochastic interpolation path from p (denominator) to q (numerator).
The log density ratio is then computed by integrating the time-score from t=0 to t=1.

Usage:
    python train_tsm_10d_tailored.py --config_path gaussians_10d_kl10/model/0

This loads the JSON config from your existing setup, uses IDENTICAL samples as other
methods via the same GAUSSIANS class, and trains a TSM model.
"""

import json
import os
import sys
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from scipy import integrate
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from __init__ import project_root
from data_handlers.gaussians import GAUSSIANS
from data_handlers.two_gaussians import kl_between_gaussians_with_mean


# ============================================================================
# TSM Model: Time Score Network
# ============================================================================

class TimeScoreNetwork(nn.Module):
    """
    MLP-based time score network for TSM.
    
    Takes input (x, t) and outputs the time-score ∂log p(x_t)/∂t.
    This is the core model for DRE-infinity / TSM.
    """
    
    def __init__(self, input_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.ELU()]
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        Args:
            x: samples, shape (batch_size, input_dim)
            t: time values, shape (batch_size, 1)
        Returns:
            time_score: ∂log p(x_t)/∂t, shape (batch_size, 1)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# ============================================================================
# TSM Dataset: Adapter for your GAUSSIANS class
# ============================================================================

class TSMGaussianDataset:
    """
    Adapter that wraps your GAUSSIANS class to work with TSM training.
    
    TSM interpolates between:
      - p(x): denominator distribution (noise), corresponds to t=0
      - q(x): numerator distribution (data), corresponds to t=1
    
    The stochastic interpolation is:
        x_t = t * q_x + sqrt(1 - t^2) * p_x
    
    This ensures same samples as your other methods when using the same seed.
    """
    
    def __init__(self, data_args, n_dims, seed, device='cpu'):
        """
        Args:
            data_args: dict with numerator_mean, numerator_cov, denominator_mean, denominator_cov
            n_dims: dimensionality
            seed: random seed for reproducibility
            device: torch device
        """
        self.n_dims = n_dims
        self.device = device
        self.seed = seed
        
        # Use YOUR GAUSSIANS class for sampling - ensures identical samples!
        self.gaussian_dataset = GAUSSIANS(
            n_samples=int(data_args["n_samples"]),
            n_dims=n_dims,
            numerator_mean=data_args["numerator_mean"],
            numerator_cov=data_args["numerator_cov"],
            denominator_mean=data_args["denominator_mean"],
            denominator_cov=data_args["denominator_cov"],
            seed=seed
        )
        
        # Store presampled data for reproducibility
        self.train_q_samples = torch.from_numpy(
            self.gaussian_dataset.trn.x.astype(np.float32)
        )
        self.val_q_samples = torch.from_numpy(
            self.gaussian_dataset.val.x.astype(np.float32)
        )
        self.test_q_samples = torch.from_numpy(
            self.gaussian_dataset.tst.x.astype(np.float32)
        )
        
        # Store true KL for evaluation
        self.true_kl = float(kl_between_gaussians_with_mean(
            np.array(data_args["numerator_mean"]),
            np.array(data_args["numerator_cov"]),
            np.array(data_args["denominator_mean"]),
            np.array(data_args["denominator_cov"])
        ))
    
    def sample_interpolation(self, batch_size, split='train', eps=1e-5):
        """
        Sample interpolated points for TSM training.
        
        Returns:
            p_samples: samples from denominator (t=0)
            q_samples: samples from numerator (t=1)
            x_t: interpolated samples
            t: time values
        """
        # Select the right data split
        if split == 'train':
            q_data = self.train_q_samples
        elif split == 'val':
            q_data = self.val_q_samples
        else:
            q_data = self.test_q_samples
        
        # Sample batch indices
        n_available = q_data.shape[0]
        idx = np.random.choice(n_available, size=batch_size, replace=True)
        q_samples = q_data[idx].to(self.device)
        
        # Sample from denominator distribution (p)
        p_samples = torch.from_numpy(
            self.gaussian_dataset.sample_denominator(batch_size).astype(np.float32)
        ).to(self.device)
        
        # Sample time t ~ Uniform(eps, 1-eps)
        t = torch.rand(batch_size, 1, device=self.device) * (1 - 2*eps) + eps
        
        # Linear interpolation: x_t = t * q + sqrt(1 - t^2) * p
        x_t = t * q_samples + torch.sqrt(1 - t**2) * p_samples
        
        return p_samples, q_samples, x_t, t
    
    def log_density_ratios(self, samples):
        """Compute true log density ratios for samples."""
        samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples
        log_p = self.gaussian_dataset.denominator_log_prob(samples_np)
        log_q = self.gaussian_dataset.numerator_log_prob(samples_np)
        return log_q - log_p


# ============================================================================
# TSM Loss Function
# ============================================================================

def tsm_time_loss(scorenet, p_samples, q_samples, x_t, t, eps=1e-5, reweight=False):
    """
    Time Score Matching loss function.
    
    This loss ensures the model learns the correct time-score without
    requiring access to the actual score function.
    
    The loss is derived from integration by parts and consists of:
    - Boundary terms at t=0 (p distribution) and t=1 (q distribution)
    - Integral terms involving the score and its derivative
    
    Note: For the Gaussian setup, we're estimating log(q/p) where:
    - p = N(0, I)  [denominator, noise] - corresponds to t=eps (near 0)
    - q = N(μ, I)  [numerator, data] - corresponds to t=1-eps (near 1)
    
    The interpolation is: x_t = t * q_sample + sqrt(1-t^2) * p_sample
    """
    device = x_t.device
    batch_size = x_t.shape[0]
    
    # Ensure t requires grad for autograd
    t = t.clone().detach().requires_grad_(True)
    
    # Boundary times
    t0 = torch.zeros((batch_size, 1), device=device) + eps
    t1 = torch.ones((batch_size, 1), device=device) - eps
    
    # Weighting functions for reweighting (optional)
    if reweight:
        lambda_t = (1 - t ** 2).squeeze()
        lambda_t0 = (1 - t0.squeeze() ** 2)
        lambda_t1 = (1 - t1.squeeze() ** 2 + eps ** 2)
        lambda_dt = (-2 * t.squeeze())
    else:
        lambda_t = lambda_t0 = lambda_t1 = 1.0
        lambda_dt = 0.0
    
    # Boundary terms
    # At t=eps (near p distribution / denominator)
    score_at_0 = scorenet(p_samples, t0)
    term1 = (2 * score_at_0).squeeze() * lambda_t0
    
    # At t=1-eps (near q distribution / numerator)
    score_at_1 = scorenet(q_samples, t1)
    term2 = (2 * score_at_1).squeeze() * lambda_t1
    
    # Score at intermediate time
    x_t_score = scorenet(x_t, t)
    
    # Derivative of score w.r.t. time (requires grad)
    x_t_score_dt = autograd.grad(
        x_t_score.sum(), t, create_graph=True
    )[0]
    
    term3 = (2 * x_t_score_dt).squeeze() * lambda_t
    term4 = x_t_score.squeeze() * lambda_dt if isinstance(lambda_dt, torch.Tensor) else x_t_score.squeeze() * lambda_dt
    term5 = (x_t_score ** 2).squeeze() * lambda_t
    
    loss = term1 - term2 + term3 + term4 + term5
    
    return loss.mean()


# ============================================================================
# Density Ratio Estimation via Integration
# ============================================================================

def compute_density_ratios(score_model, samples, eps=1e-5, rtol=1e-6, atol=1e-6):
    """
    Compute log density ratios by integrating the time-score from eps to 1.
    
    log(q(x)/p(x)) = ∫_{eps}^{1} ∂log p(x_t)/∂t dt
    
    Args:
        score_model: trained TSM model
        samples: samples to evaluate, shape (n_samples, dim)
        eps: small value to avoid numerical issues at boundaries
    
    Returns:
        log_ratios: estimated log density ratios
        nfe: number of function evaluations used by ODE solver
    """
    score_model.eval()
    device = next(score_model.parameters()).device
    
    with torch.no_grad():
        def ode_func(t, y, samples):
            """ODE function for integration."""
            n = samples.size(0)
            t_tensor = torch.ones(n, 1, device=device) * t
            score = score_model(samples.to(device), t_tensor)
            return score.squeeze().cpu().numpy()
        
        # Partial function with fixed samples
        samples_tensor = torch.from_numpy(samples.astype(np.float32)) if isinstance(samples, np.ndarray) else samples
        ode_fn = partial(ode_func, samples=samples_tensor)
        
        # Integrate from eps to 1
        solution = integrate.solve_ivp(
            ode_fn,
            (eps, 1.0),
            np.zeros(samples_tensor.shape[0]),
            method='RK45',
            rtol=rtol,
            atol=atol
        )
        
        log_ratios = solution.y[:, -1]
        nfe = solution.nfev
        
    return log_ratios, nfe


# ============================================================================
# Training Function
# ============================================================================

def train_tsm(config, dataset, save_dir, device='cpu'):
    """
    Train a TSM model.
    
    Args:
        config: training configuration dict
        dataset: TSMGaussianDataset instance
        save_dir: directory to save model and logs
        device: torch device
    
    Returns:
        model: trained TSM model
        metrics: dict of training metrics
    """
    # Create model
    model = TimeScoreNetwork(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['n_epochs'],
        eta_min=config['lr'] * 0.01
    )
    
    # Training loop
    best_kl_error = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    metrics = {'train_losses': [], 'val_losses': [], 'val_kl_errors': []}
    
    for epoch in range(config['n_epochs']):
        # Training
        model.train()
        epoch_losses = []
        
        n_batches = max(1, config['n_train_samples'] // config['batch_size'])
        for _ in range(n_batches):
            p_samples, q_samples, x_t, t = dataset.sample_interpolation(
                config['batch_size'], split='train'
            )
            
            optimizer.zero_grad()
            loss = tsm_time_loss(
                model, p_samples, q_samples, x_t, t,
                reweight=config.get('reweight', False)
            )
            loss.backward()
            
            # Gradient clipping
            if config.get('grad_clip', -1) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_losses)
        metrics['train_losses'].append(avg_train_loss)
        
        # Validation (note: we need gradients for tsm_time_loss even in eval mode)
        model.eval()
        p_val, q_val, x_t_val, t_val = dataset.sample_interpolation(
            min(config['batch_size'], len(dataset.val_q_samples)),
            split='val'
        )
        with torch.no_grad():
            # For validation, compute a simpler loss that doesn't require gradients
            # Use the time-score at boundaries as a proxy
            t0 = torch.zeros((p_val.shape[0], 1), device=p_val.device) + 1e-5
            t1 = torch.ones((q_val.shape[0], 1), device=q_val.device) - 1e-5
            score_at_0 = model(p_val, t0).squeeze()
            score_at_1 = model(q_val, t1).squeeze()
            # Simple boundary consistency: at t=0, score should be negative (going from p to q)
            # at t=1, score should be positive
            val_loss = (score_at_0.pow(2).mean() + score_at_1.pow(2).mean()).item()
        metrics['val_losses'].append(val_loss)
        
        # Evaluate KL estimation periodically
        if (epoch + 1) % config.get('eval_freq', 50) == 0:
            val_samples = dataset.val_q_samples[:min(500, len(dataset.val_q_samples))]
            est_log_ratios, _ = compute_density_ratios(model, val_samples.numpy())
            est_kl = np.mean(est_log_ratios)
            kl_error = abs(est_kl - dataset.true_kl) / dataset.true_kl
            metrics['val_kl_errors'].append((epoch, kl_error, est_kl))
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"est_KL={est_kl:.2f}, true_KL={dataset.true_kl:.2f}, rel_err={kl_error:.4f}")
            
            # Save best model based on KL error (when we have KL measurements)
            if kl_error < best_kl_error:
                best_kl_error = kl_error
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'kl_error': kl_error,
                    'est_kl': est_kl
                }, os.path.join(save_dir, 'tsm_model_best.pt'))
                print(f"  -> New best model saved (KL error: {kl_error:.4f})")
            else:
                patience_counter += 1
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping based on KL error (only increment counter during KL eval)
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, os.path.join(save_dir, 'tsm_model_final.pt'))
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    # Load best model for return
    checkpoint = torch.load(os.path.join(save_dir, 'tsm_model_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, metrics


# ============================================================================
# Config and Main
# ============================================================================

def _load_tdre_config(config_path):
    """Load config from your existing JSON format."""
    cfg_path = os.path.join(project_root, "configs", f"{config_path}.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find config at {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _extract_stub(save_dir):
    return os.path.basename(os.path.normpath(save_dir))


def _ensure_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, list):
        return [_ensure_serializable(x) for x in data]
    if isinstance(data, dict):
        return {k: _ensure_serializable(v) for k, v in data.items()}
    return data


def make_tsm_config(args, n_dims, n_train_samples):
    """Create TSM training configuration."""
    batch_size = min(args.batch_size, n_train_samples)
    batch_size = max(batch_size, 8)
    
    config = {
        "input_dim": n_dims,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_train_samples": n_train_samples,
        "batch_size": batch_size,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "reweight": args.reweight,
        "eval_freq": args.eval_freq,
    }
    return config


def save_metadata(save_dir, original_config, tsm_config, true_kl):
    """Save metadata for evaluation."""
    metadata = {
        "model_type": "tsm",
        "dataset_name": original_config["data"]["dataset_name"],
        "data_seed": original_config["data"]["data_seed"],
        "data_args": original_config["data"]["data_args"],
        "frac": original_config["data"].get("frac", 1.0),
        "true_kl": true_kl,
        "training_config": tsm_config
    }
    metadata = _ensure_serializable(metadata)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def parse_args():
    parser = ArgumentParser(
        description="Train TSM models for gaussians_10d configs",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path like gaussians_10d_kl10/model/0")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="TSM MLP hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of hidden layers in TSM network")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=-1.0,
                        help="Gradient clipping norm. Use -1 or 0 to disable (default: disabled)")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--reweight", action="store_true",
                        help="Use reweighted TSM loss")
    parser.add_argument("--eval_freq", type=int, default=50,
                        help="How often to evaluate KL estimation")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Extra offset added to data_seed")
    parser.add_argument("--save_root", type=str, default="tsm_gaussians_10d_kl",
                        help="Subdirectory prefix inside saved_models/ (KL value will be appended)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    original_config = _load_tdre_config(args.config_path)
    data_cfg = original_config["data"]
    data_args = data_cfg["data_args"]
    n_dims = int(data_cfg["n_dims"])
    seed = int(data_cfg.get("data_seed", 0)) + args.seed_offset
    
    # Determine KL value for save directory from config path
    # Extract KL from path like "gaussians_10d_kl10/model/0" or "gaussians_pstar_p0_kl10/model/0"
    import re
    kl_match = re.search(r'_kl(\d+)', args.config_path)
    if kl_match:
        target_kl = int(kl_match.group(1))
    else:
        # Fallback to config value
        target_kl = int(data_args.get("target_kl", data_args.get("analytic_kl", 10)))
    save_root = f"{args.save_root}{target_kl}"
    
    # Create dataset
    dataset = TSMGaussianDataset(
        data_args=data_args,
        n_dims=n_dims,
        seed=seed,
        device=device
    )
    
    n_train_samples = len(dataset.train_q_samples)
    tsm_config = make_tsm_config(args, n_dims, n_train_samples)
    
    # Determine save directory
    stub = _extract_stub(data_cfg["save_dir"])
    # Transform stub from tdre format to tsm format
    # e.g., "kl10_0" stays as "kl10_0"
    save_dir = os.path.join(project_root, "saved_models", save_root, stub)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nTraining TSM for config {args.config_path}")
    print(f"  Training samples: {n_train_samples}")
    print(f"  Dimensions: {n_dims}")
    print(f"  True KL: {dataset.true_kl:.2f}")
    print(f"  Saving to: {save_dir}")
    
    # Train model
    model, metrics = train_tsm(tsm_config, dataset, save_dir, device)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_samples = dataset.test_q_samples[:min(1000, len(dataset.test_q_samples))]
    est_log_ratios, nfe = compute_density_ratios(model, test_samples.numpy())
    est_kl = np.mean(est_log_ratios)
    rel_error = abs(est_kl - dataset.true_kl) / dataset.true_kl
    print(f"  Estimated KL: {est_kl:.4f}")
    print(f"  True KL: {dataset.true_kl:.4f}")
    print(f"  Relative Error: {rel_error:.4f}")
    print(f"  ODE solver used {nfe} function evaluations")
    
    # Save metadata
    save_metadata(save_dir, original_config, tsm_config, dataset.true_kl)
    print("\nDone!")


if __name__ == "__main__":
    main()
