#!/usr/bin/env python
"""
Train TSM model using the JOINT score network approach from dre-infinity.
This learns BOTH the data score (∇_x log p) and time score (∂log p/∂t) jointly.

Key differences from time-only approach:
- Model outputs [score_x, score_t] instead of just score_t
- Loss includes SSM (Sliced Score Matching) for data score + time loss
- Uses VPSDE marginal_prob to scale the data score

Compatible with your existing configs for fair comparison.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from scipy import integrate
from functools import partial

# Get project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = _this_dir

sys.path.insert(0, project_root)
from data_handlers.gaussians import GAUSSIANS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# VPSDE: For computing marginal_prob (used to scale data score)
# Exact copy from dre-infinity-main/sde_lib.py
# ============================================================================
class VPSDE:
    """Variance Preserving SDE from dre-infinity."""
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def marginal_prob(self, x, t):
        """Compute mean and std of p(x_t | x_0)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if len(x.size()) < 4:
            mean = torch.exp(log_mean_coeff[:, None]) * x
        else:
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


# ============================================================================
# MODEL: JointScoreNetwork - Exact copy from dre-infinity
# Learns both data score and time score
# ============================================================================
class JointScoreNetwork(nn.Module):
    """
    Joint MLP-based score network that outputs BOTH:
    - out_x: data score ∇_x log p(x_t), scaled by std
    - out_t: time score ∂log p(x_t)/∂t
    
    Exact architecture from dre-infinity's toy_joint_scorenet.
    """
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.sde = VPSDE()
        
        # Time score head
        self.time = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1)
        )
        
        # Data score head
        self.score = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, in_dim)
        )
        
        # Shared backbone
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim * 2),
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        h = self.net(xt)
        h_x, h_t = torch.chunk(h, 2, dim=1)
        out_t = self.time(h_t)
        out_x = self.score(h_x)
        
        # Scale data score by standard deviation (from VPSDE)
        _, std = self.sde.marginal_prob(x, t)
        out_x = out_x / std
        
        return [out_x, out_t]


# ============================================================================
# LOSS: toy_joint_score_estimation - Exact copy from dre-infinity
# Combines SSM (for data score) + time loss
# ============================================================================
def toy_joint_score_estimation(scorenet, samples, t, eps=1e-5, likelihood_weighting=False):
    """
    Joint score estimation loss from dre-infinity.
    
    Combines:
    1. SSM (Sliced Score Matching) for data score ∇_x log p
    2. Time loss for time score ∂log p/∂t
    
    Args:
        scorenet: JointScoreNetwork model
        samples: tuple of (px, qx, xt) - noise samples, data samples, interpolated
        t: time values
        eps: small value for numerical stability
        likelihood_weighting: whether to use λ(t) = 1 - t²
    
    Returns:
        loss: combined SSM + time loss
    """
    px, qx, xt = samples
    px = px.to(device)
    qx = qx.to(device)
    xt = xt.to(device)
    t = t.to(device)
    
    # Boundary times
    t0 = torch.zeros((len(px), 1)).to(px.device) + eps
    t1 = torch.ones((len(qx), 1)).to(qx.device) - eps
    
    # Get scores - need gradients for SSM
    xt.requires_grad_(True)
    vectors = torch.randn_like(xt, device=xt.device)
    score_x, score_t = scorenet(xt, t)
    
    # Compute SSM loss components
    grad1 = torch.cat([score_x, score_t], dim=-1)
    gradv = torch.sum(score_x * vectors)
    grad2 = autograd.grad(gradv, xt, create_graph=True)[0]
    
    # Weighting functions
    if likelihood_weighting:
        lambda_t = (1 - t ** 2).squeeze()
        lambda_t0 = (1 - t0.squeeze() ** 2)
        lambda_t1 = (1 - t1.squeeze() ** 2 + eps ** 2)
        lambda_dt = (-2 * t.squeeze())
    else:
        lambda_t = lambda_t0 = lambda_t1 = 1
        lambda_dt = 0
    
    # SSM loss: ||s(x,t)||² / 2 + ∇·s(x,t)
    ssm_loss1 = (torch.sum(grad1 * grad1, dim=-1) / 2.).view(
        lambda_t.size() if isinstance(lambda_t, torch.Tensor) else -1) * lambda_t
    ssm_loss2 = torch.sum(vectors * grad2, dim=-1).view(
        lambda_t.size() if isinstance(lambda_t, torch.Tensor) else -1) * lambda_t
    ssm_loss = ssm_loss1 + ssm_loss2
    
    # Time loss: boundary terms + ∂score_t/∂t
    term1 = (scorenet(px, t0)[-1]).squeeze() * lambda_t0  # t=0 boundary
    term2 = (scorenet(qx, t1)[-1]).squeeze() * lambda_t1  # t=1 boundary
    
    # Derivative of time score w.r.t. t
    with torch.enable_grad():
        t_for_grad = t.detach().clone().requires_grad_(True)
        _, score_t_for_grad = scorenet(xt.detach(), t_for_grad)
        xt_score_dt = autograd.grad(score_t_for_grad.sum(), t_for_grad, create_graph=True)[0]
    
    term3 = xt_score_dt.squeeze() * lambda_t
    term4 = score_t.squeeze() * lambda_dt if isinstance(lambda_dt, torch.Tensor) else score_t.squeeze() * lambda_dt
    
    time_loss = term1 - term2 + term3 + term4
    
    # Combined loss
    loss = ssm_loss + time_loss
    
    return loss.mean()


# ============================================================================
# DENSITY RATIO ESTIMATION: Using time score output
# ============================================================================
def get_toy_density_ratio_fn(rtol=1e-6, atol=1e-6, method='RK45', eps=1e-5):
    """
    Compute density ratios by integrating the TIME score from eps to 1.
    For joint model, we use the [-1] (second) output which is the time score.
    """
    def ratio_fn(score_model, x, score_type='joint'):
        with torch.no_grad():
            def ode_func(t, y, x, score_model):
                score_model.eval()
                t_tensor = (torch.ones(x.size(0)) * t).to(x.device).view(-1, 1)
                
                if score_type == 'joint':
                    # Joint model returns [score_x, score_t], use score_t
                    rx = score_model(x, t_tensor)[-1]
                else:
                    rx = score_model(x, t_tensor)
                rx = np.reshape(rx.detach().cpu().numpy(), -1)
                return rx
            
            p_get_rx = partial(ode_func, x=x, score_model=score_model)
            solution = integrate.solve_ivp(p_get_rx, (eps, 1.),
                                           np.zeros((x.shape[0],)),
                                           method=method, rtol=rtol, atol=atol)
            nfe = solution.nfev
            density_ratio = solution.y[:, -1]
            
            return density_ratio, nfe
    
    return ratio_fn


# ============================================================================
# DATA HANDLING: Same as other TSM scripts
# ============================================================================
class TSMGaussianDataset:
    """Wraps GAUSSIANS class for TSM training."""
    
    def __init__(self, data_args, n_dims=10, seed=None):
        self.n_dims = n_dims
        self.seed = seed
        
        self.gaussian_data = GAUSSIANS(
            n_samples=data_args['n_samples'],
            n_dims=n_dims,
            numerator_mean=data_args['numerator_mean'],
            numerator_cov=data_args['numerator_cov'],
            denominator_mean=data_args['denominator_mean'],
            denominator_cov=data_args['denominator_cov'],
            seed=seed
        )
        
        self.train_q_samples = torch.tensor(self.gaussian_data.trn.x.astype(np.float32))
        self.val_q_samples = torch.tensor(self.gaussian_data.val.x.astype(np.float32))
        
        self.true_kl = data_args.get('true_kl', None)
        self.n_train = len(self.train_q_samples)
        self.n_val = len(self.val_q_samples)
    
    def sample_batch(self, batch_size, t, device='cpu'):
        """Sample batch for training: (px, qx, xt)."""
        q_idx = torch.randint(0, self.n_train, (batch_size,))
        qx = self.train_q_samples[q_idx].to(device)
        
        # px from standard Gaussian
        px = torch.randn(batch_size, self.n_dims, device=device)
        
        # Interpolated samples: x_t = t * qx + sqrt(1-t²) * noise
        noise = torch.randn_like(qx)
        mean = qx * t
        std = torch.sqrt(1 - t ** 2)
        xt = mean + std * noise
        
        return (px, qx, xt)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_tsm_joint(config_path, n_epochs=1000, batch_size=512, lr=1e-3,
                    hidden_dim=256, eval_freq=100, patience=10,
                    likelihood_weighting=False, weight_decay=0.0,
                    save_root=None, device_str='cpu'):
    """Train joint TSM model."""
    
    global device
    device = device_str
    
    # Load config
    cfg_path = os.path.join(project_root, 'configs', f'{config_path}.json')
    with open(cfg_path) as f:
        config = json.load(f)
    
    data_cfg = config['data']
    data_args = data_cfg['data_args']
    
    data_args_converted = {
        'numerator_mean': data_args['numerator_mean'],
        'numerator_cov': data_args['numerator_cov'],
        'denominator_mean': data_args['denominator_mean'],
        'denominator_cov': data_args['denominator_cov'],
        'n_samples': data_args['n_samples'],
        'true_kl': data_args.get('analytic_kl', data_args.get('true_mutual_info', None)),
    }
    
    data_seed = data_cfg.get('data_seed', 42)
    n_dims = int(data_cfg.get('n_dims', len(data_args_converted['numerator_mean'])))
    dataset = TSMGaussianDataset(data_args_converted, n_dims=n_dims, seed=data_seed)
    
    print(f"\nTraining TSM (JOINT) for config {config_path}")
    print(f"  Training samples: {dataset.n_train}")
    print(f"  Dimensions: {n_dims}")
    print(f"  True KL: {dataset.true_kl:.2f}")
    print(f"  Likelihood weighting: {likelihood_weighting}")
    
    # Create JOINT model
    model = JointScoreNetwork(in_dim=n_dims, h_dim=hidden_dim)
    model = model.to(device)
    
    # Optimizer with optional weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup save directory
    if save_root is None:
        save_root = f"tsm_joint_gaussians_10d_kl{int(dataset.true_kl)}"
    
    config_num = config_path.split('/')[-1]
    save_dir = os.path.join(project_root, 'saved_models', save_root, f"kl{int(dataset.true_kl)}_{config_num}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"  Saving to: {save_dir}")
    
    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({
            'config_path': config_path,
            'data_args': data_args_converted,
            'data_seed': data_seed,
            'training_config': {
                'model_type': 'joint',
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'input_dim': n_dims,
                'likelihood_weighting': likelihood_weighting,
                'weight_decay': weight_decay,
            }
        }, f, indent=2)
    
    # Training
    eps = 1e-5
    best_kl_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    density_ratio_fn = get_toy_density_ratio_fn(eps=eps)
    metrics = {'train_losses': [], 'val_kl_errors': []}
    
    for epoch in range(n_epochs):
        model.train()
        
        # Sample batch
        t = torch.rand(batch_size, 1) * (1 - eps) + eps
        t = t.to(device)
        samples = dataset.sample_batch(batch_size, t, device=device)
        
        # Compute joint loss
        optimizer.zero_grad()
        loss = toy_joint_score_estimation(
            model, samples, t, eps=eps, likelihood_weighting=likelihood_weighting
        )
        loss.backward()
        
        # Gradient clipping for stability (joint training can be unstable)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        metrics['train_losses'].append(loss.item())
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
        
        # Evaluation
        if (epoch + 1) % eval_freq == 0:
            model.eval()
            
            val_samples = dataset.val_q_samples[:1000].to(device)
            log_ratios, nfe = density_ratio_fn(model, val_samples, score_type='joint')
            # Negate: TSM integration convention gives log(p/q), we need log(q/p)
            est_kl = -np.mean(log_ratios)
            rel_err = abs(est_kl - dataset.true_kl) / dataset.true_kl
            
            metrics['val_kl_errors'].append((epoch, rel_err, est_kl))
            print(f"  Eval: est_KL={est_kl:.4f}, true_KL={dataset.true_kl:.4f}, "
                  f"rel_err={rel_err:.4f}, NFE={nfe}")
            
            # Early stopping based on KL error
            if rel_err < best_kl_error:
                best_kl_error = rel_err
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'kl_error': rel_err,
                    'est_kl': est_kl,
                    'true_kl': dataset.true_kl,
                }, os.path.join(save_dir, 'tsm_model_best.pt'))
                print(f"  -> New best model saved (KL error: {best_kl_error:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Save final model and metrics
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_kl_error': best_kl_error,
    }, os.path.join(save_dir, 'tsm_model_final.pt'))
    
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nTraining complete!")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best KL error: {best_kl_error:.4f}")
    
    return save_dir, best_kl_error


def main():
    parser = argparse.ArgumentParser(description='Train TSM using JOINT score network')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config (relative to configs/)')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--reweight', action='store_true',
                        help='Use likelihood weighting')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    train_tsm_joint(
        config_path=args.config_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        eval_freq=args.eval_freq,
        patience=args.patience,
        likelihood_weighting=args.reweight,
        weight_decay=args.weight_decay,
        save_root=args.save_root,
        device_str=args.device,
    )


if __name__ == '__main__':
    main()

