#!/usr/bin/env python
"""
Train TSM model using the ORIGINAL dre-infinity codebase components.
This uses their exact model architecture, loss function, and evaluation.

The only thing different is that we use our configs for data generation
to ensure fair comparison with other DRE methods.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from scipy import integrate
from functools import partial

# Get project root BEFORE any sys.path modifications
_this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = _this_dir

# Import our data handler
sys.path.insert(0, project_root)
from data_handlers.gaussians import GAUSSIANS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# MODEL: Exact copy from dre-infinity-main/models/toy_networks.py (toy_time_scorenet)
# ============================================================================
class TimeScoreNetwork(nn.Module):
    """
    Simple MLP-based score network (for toy gaussian problems)
    EXACT COPY from dre-infinity codebase.
    """
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim + 1, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, 1),
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        h = self.net(xt)
        return h


# ============================================================================
# SDE: Exact copy of marginal_prob from ToyInterpXt
# ============================================================================
class ToyInterpXt:
    """
    Linear interpolation SDE from dre-infinity.
    x_t = t * q + sqrt(1 - t^2) * p, where p ~ N(0, I)
    """
    @staticmethod
    def marginal_prob(x, t):
        """
        Returns mean and std for the marginal distribution at time t.
        mean = x * t
        std = sqrt(1 - t^2)
        """
        std = torch.sqrt(1 - t**2)
        mean = x * t
        return mean, std
    
    @staticmethod
    def prior_sampling(shape, device='cpu'):
        """Sample from prior p ~ N(0, I)"""
        return torch.randn(*shape, device=device)


# ============================================================================
# LOSS: Exact copy from dre-infinity-main/toy_losses.py (toy_timewise_score_estimation)
# ============================================================================
def toy_timewise_score_estimation(scorenet, samples, t, eps=1e-5, likelihood_weighting=False):
    """
    EXACT COPY from dre-infinity codebase.
    
    in objective, T = [0, 1]
    px, qx, xt: (batch_size, dim)
    t: (batch_size, 1)

    we are reweighting the output of the score network (most recent version)
    """
    px, qx, xt = samples
    px = px.to(device)
    qx = qx.to(device)
    xt = xt.to(device)
    t = t.to(device)

    # reweighted version
    t0 = torch.zeros((len(px), 1)).to(px.device) + eps
    t1 = torch.ones((len(qx), 1)).to(qx.device)

    if likelihood_weighting:
        lambda_t = (1 - t ** 2).squeeze()
        lambda_t0 = (1 - t0.squeeze() ** 2)
        lambda_t1 = (1 - t1.squeeze() ** 2 + eps ** 2)
        lambda_dt = (-2 * t.squeeze())
    else:
        lambda_t = lambda_t0 = lambda_t1 = 1
        lambda_dt = 0

    term1 = (2 * scorenet(px, t0)).squeeze() * lambda_t0
    term2 = (2 * scorenet(qx, t1)).squeeze() * lambda_t1

    # need to differentiate score wrt t
    t.requires_grad_(True)
    xt_score = scorenet(xt, t)  # dim = 1
    xt_score_dt = autograd.grad(xt_score.sum(), t, create_graph=True)[0]
    term3 = (2 * xt_score_dt).squeeze() * lambda_t
    term4 = (xt_score).squeeze() * lambda_dt
    term5 = (xt_score ** 2).squeeze() * lambda_t

    loss = term1 - term2 + term3 + term4 + term5

    # 1-d so we can just take the mean rather than summing
    return loss.mean(), term3.mean(), term4.mean(), term5.mean(), term1.mean(), term2.mean()


# ============================================================================
# DENSITY RATIO ESTIMATION: Exact copy from dre-infinity-main/density_ratios.py
# ============================================================================
def get_toy_density_ratio_fn(rtol=1e-6, atol=1e-6, method='RK45', eps=1e-5):
    """
    EXACT COPY from dre-infinity codebase.
    Create a function to compute the density ratios of a given point.
    """
    def ratio_fn(score_model, x, score_type='time'):
        with torch.no_grad():
            def ode_func(t, y, x, score_model):
                score_model.eval()
                t_tensor = (torch.ones(x.size(0)) * t).to(x.device).view(-1, 1)
                x = x.to(x.device)

                if score_type == 'joint':
                    rx = score_model(x, t_tensor)[-1]
                else:
                    rx = score_model(x, t_tensor)
                rx = np.reshape(rx.detach().cpu().numpy(), -1)

                return rx

            # now just a function of t
            p_get_rx = partial(ode_func, x=x, score_model=score_model)
            # Integrate from eps to 1 (following their toy dataset convention)
            solution = integrate.solve_ivp(p_get_rx, (eps, 1.),
                                           np.zeros((x.shape[0],)),
                                           method=method, rtol=rtol, atol=atol)
            nfe = solution.nfev
            density_ratio = solution.y[:, -1]

            return density_ratio, nfe

    return ratio_fn


# ============================================================================
# DATA HANDLING: Wrapper for our Gaussians dataset
# ============================================================================
class TSMGaussianDataset:
    """
    Wraps our GAUSSIANS class for use with TSM training.
    Matches the interface expected by dre-infinity's training loop.
    """
    def __init__(self, data_args, n_dims=10, seed=None):
        self.data_args = data_args
        self.n_dims = n_dims
        self.seed = seed
        
        # Create GAUSSIANS instance using the same interface as our other methods
        self.gaussian_data = GAUSSIANS(
            n_samples=data_args['n_samples'],
            n_dims=n_dims,
            numerator_mean=data_args['numerator_mean'],
            numerator_cov=data_args['numerator_cov'],
            denominator_mean=data_args['denominator_mean'],
            denominator_cov=data_args['denominator_cov'],
            seed=seed
        )
        
        # Get samples as torch tensors
        # In TSM convention: qx = data (numerator), px = noise (we sample fresh each batch)
        self.train_q_samples = torch.tensor(self.gaussian_data.trn.x.astype(np.float32))
        self.val_q_samples = torch.tensor(self.gaussian_data.val.x.astype(np.float32))
        
        self.true_kl = data_args.get('true_kl', None)
        self.n_train = len(self.train_q_samples)
        self.n_val = len(self.val_q_samples)
        
    def sample_batch(self, batch_size, t, device='cpu'):
        """
        Sample a batch for TSM training.
        Returns (px, qx, xt) and t following dre-infinity convention.
        
        px: samples from p (reference/prior) - used at t=1 boundary
        qx: samples from q (data) - used at t=0 boundary  
        xt: interpolated samples for interior loss terms
        """
        # Random indices for q samples
        q_idx = torch.randint(0, self.n_train, (batch_size,))
        qx = self.train_q_samples[q_idx].to(device)
        
        # Sample px from standard Gaussian (reference distribution)
        # In dre-infinity's ToyInterpXt: p is N(0,I)
        px = torch.randn(batch_size, self.n_dims, device=device)
        
        # Create interpolated samples using ToyInterpXt marginal
        # x_t = t * qx + sqrt(1-t^2) * noise
        mean, std = ToyInterpXt.marginal_prob(qx, t)
        noise = torch.randn_like(qx)
        xt = mean + std * noise
        
        return (px, qx, xt)


def train_tsm_original(config_path, n_epochs=1000, batch_size=512, lr=1e-3,
                       hidden_dim=256, eval_freq=100, patience=10,
                       likelihood_weighting=False, save_root=None, device='cpu'):
    """
    Train TSM using the original dre-infinity components.
    """
    # Load config from JSON file
    cfg_path = os.path.join(project_root, 'configs', f'{config_path}.json')
    with open(cfg_path) as f:
        config = json.load(f)
    
    # Extract data args from the nested structure
    data_cfg = config['data']
    data_args = data_cfg['data_args']
    
    # Use the same format as the original config
    data_args_converted = {
        'numerator_mean': data_args['numerator_mean'],
        'numerator_cov': data_args['numerator_cov'],
        'denominator_mean': data_args['denominator_mean'],
        'denominator_cov': data_args['denominator_cov'],
        'n_samples': data_args['n_samples'],
        'true_kl': data_args.get('analytic_kl', data_args.get('true_mutual_info', None)),
    }
    
    # Create dataset
    data_seed = data_cfg.get('data_seed', 42)
    n_dims = int(data_cfg.get('n_dims', len(data_args_converted['numerator_mean'])))
    dataset = TSMGaussianDataset(data_args_converted, n_dims=n_dims, seed=data_seed)
    
    print(f"\nTraining TSM (ORIGINAL) for config {config_path}")
    print(f"  Training samples: {dataset.n_train}")
    print(f"  Dimensions: {n_dims}")
    print(f"  True KL: {dataset.true_kl:.2f}")
    print(f"  Likelihood weighting: {likelihood_weighting}")
    
    # Create model - EXACT architecture from dre-infinity
    model = TimeScoreNetwork(in_dim=n_dims, h_dim=hidden_dim)
    model = model.to(device)
    
    # Optimizer - following their setup
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    # Setup save directory
    if save_root is None:
        save_root = f"tsm_original_gaussians_10d_kl{int(dataset.true_kl)}"
    
    # Extract config number from path
    config_num = config_path.split('/')[-1]
    save_dir = os.path.join(project_root, 'saved_models', save_root, f"kl{int(dataset.true_kl)}_{config_num}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"  Saving to: {save_dir}")
    
    # Save config for reproducibility
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({
            'config_path': config_path,
            'data_args': data_args_converted,
            'data_seed': data_seed,
            'training_config': {
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'input_dim': n_dims,
                'likelihood_weighting': likelihood_weighting,
            }
        }, f, indent=2)
    
    # Training
    eps = 1e-5
    best_kl_error = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    # Density ratio function for evaluation
    density_ratio_fn = get_toy_density_ratio_fn(eps=eps)
    
    for epoch in range(n_epochs):
        model.train()
        
        # Sample batch
        t = torch.rand(batch_size, 1) * (1 - eps)
        t = t.to(device)
        samples = dataset.sample_batch(batch_size, t, device=device)
        
        # Compute loss using EXACT dre-infinity function
        optimizer.zero_grad()
        loss, term3, term4, term5, term1, term2 = toy_timewise_score_estimation(
            model, samples, t, eps=eps, likelihood_weighting=likelihood_weighting
        )
        loss.backward()
        optimizer.step()
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
        
        # Evaluation
        if (epoch + 1) % eval_freq == 0:
            model.eval()
            
            # Compute KL estimate on validation set
            val_samples = dataset.val_q_samples[:1000].to(device)
            log_ratios, nfe = density_ratio_fn(model, val_samples, score_type='time')
            # Negate: TSM integration convention gives log(p/q), we need log(q/p)
            est_kl = -np.mean(log_ratios)
            rel_err = abs(est_kl - dataset.true_kl) / dataset.true_kl
            kl_error = rel_err
            
            print(f"  Eval: est_KL={est_kl:.4f}, true_KL={dataset.true_kl:.4f}, "
                  f"rel_err={rel_err:.4f}, NFE={nfe}")
            
            # Early stopping based on KL error
            if kl_error < best_kl_error:
                best_kl_error = kl_error
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'kl_error': kl_error,
                    'est_kl': est_kl,
                    'true_kl': dataset.true_kl,
                }, os.path.join(save_dir, 'tsm_model_best.pt'))
                print(f"  -> New best model saved (KL error: {best_kl_error:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_epoch': best_epoch,
        'best_kl_error': best_kl_error,
    }, os.path.join(save_dir, 'tsm_model_final.pt'))
    
    print(f"\nTraining complete!")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best KL error: {best_kl_error:.4f}")
    
    return save_dir, best_kl_error


def main():
    parser = argparse.ArgumentParser(description='Train TSM using original dre-infinity components')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config directory (relative to configs/)')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (their default is smaller, but we use 256 for fairness)')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--reweight', action='store_true',
                        help='Use likelihood weighting (their λ(t)=1-t²)')
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    train_tsm_original(
        config_path=args.config_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        eval_freq=args.eval_freq,
        patience=args.patience,
        likelihood_weighting=args.reweight,
        save_root=args.save_root,
        device=args.device,
    )


if __name__ == '__main__':
    main()

