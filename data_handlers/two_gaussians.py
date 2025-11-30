"""
Two Gaussian Distributions for Density Ratio Estimation Comparison

This module defines two 10-dimensional multivariate Gaussian distributions
P0 and P1 for comparing BDRE and TDRE methods.
"""

import numpy as np
from scipy.stats import multivariate_normal


def kl_between_gaussians_with_mean(mu1, cov1, mu0, cov0):
    """
    Compute KL divergence between two Gaussian distributions with non-zero means.
    KL(P1 || P0) where P1 ~ N(mu1, cov1) and P0 ~ N(mu0, cov0)
    
    Formula:
    KL(P1||P0) = 0.5 * (tr(Σ0^{-1}Σ1) + (μ0-μ1)^T Σ0^{-1} (μ0-μ1) - d + log(det(Σ0)/det(Σ1)))
    """
    d = len(mu0)
    
    # Compute trace term: tr(Σ0^{-1}Σ1)
    trace_term = np.trace(np.linalg.solve(cov0, cov1))
    
    # Compute quadratic term: (μ0-μ1)^T Σ0^{-1} (μ0-μ1)
    mu_diff = mu0 - mu1
    quad_term = mu_diff.T @ np.linalg.solve(cov0, mu_diff)
    
    # Compute log determinant term: log(det(Σ0)/det(Σ1))
    log_det_cov0 = np.linalg.slogdet(cov0)[1]
    log_det_cov1 = np.linalg.slogdet(cov1)[1]
    log_det_ratio = log_det_cov0 - log_det_cov1
    
    # Combine all terms
    kl = 0.5 * (trace_term + quad_term - d + log_det_ratio)
    
    return kl


class TwoGaussians:
    """
    Dataset with two different multivariate Gaussian distributions P0 and P1.
    Used for density ratio estimation experiments.
    """
    
    class Data:
        """Wrapper for dataset splits"""
        def __init__(self, data_p0, data_p1):
            self.x_p0 = data_p0  # Samples from P0 (denominator)
            self.x_p1 = data_p1  # Samples from P1 (numerator)
            self.N_p0 = data_p0.shape[0]
            self.N_p1 = data_p1.shape[0]
            
    def __init__(self, n_samples, n_dims=10, mean_shift=2.0, cov_scale=1.0,
                 seed=None, mu0=None, mu1=None, cov0=None, cov1=None):
        """
        Initialize two Gaussian distributions P0 and P1.
        
        Args:
            n_samples: Number of samples to generate for each distribution
            n_dims: Dimensionality (default=10)
            mean_shift: How much to shift P1 mean from P0 (default=2.0)
            cov_scale: Scaling factor for P1 covariance (default=1.0)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.n_dims = n_dims
        self.n_samples = n_samples
        
        # Define P0: default N(0, I) unless custom parameters supplied
        if mu0 is not None:
            self.mu0 = np.asarray(mu0, dtype=np.float32)
        else:
            self.mu0 = np.zeros(n_dims, dtype=np.float32)
        if cov0 is not None:
            self.cov0 = np.asarray(cov0, dtype=np.float32)
        else:
            self.cov0 = np.eye(n_dims, dtype=np.float32)
        
        # Define P1: default shifted/scaled Gaussian unless overridden
        if mu1 is not None:
            self.mu1 = np.asarray(mu1, dtype=np.float32)
        else:
            self.mu1 = np.zeros(n_dims, dtype=np.float32)
            self.mu1[:min(3, n_dims)] = mean_shift  # Shift first 3 dimensions
        if cov1 is not None:
            self.cov1 = np.asarray(cov1, dtype=np.float32)
        else:
            self.cov1 = cov_scale * np.eye(n_dims, dtype=np.float32)
            if n_dims >= 2:
                for i in range(min(3, n_dims-1)):
                    self.cov1[i, i+1] = 0.3
                    self.cov1[i+1, i] = 0.3
        
        # Create distributions
        self.dist_p0 = multivariate_normal(mean=self.mu0, cov=self.cov0)
        self.dist_p1 = multivariate_normal(mean=self.mu1, cov=self.cov1)
        
        # Compute true KL divergence analytically
        self.true_kl = kl_between_gaussians_with_mean(self.mu1, self.cov1, self.mu0, self.cov0)
        
        # Generate train/val/test splits
        self.trn = self.Data(
            self.sample_p0(n_samples),
            self.sample_p1(n_samples)
        )
        self.val = self.Data(
            self.sample_p0(n_samples),
            self.sample_p1(n_samples)
        )
        self.tst = self.Data(
            self.sample_p0(n_samples),
            self.sample_p1(n_samples)
        )
        
    def sample_p0(self, n):
        """Sample from P0 (denominator distribution)"""
        return self.dist_p0.rvs(n)
    
    def sample_p1(self, n):
        """Sample from P1 (numerator distribution)"""
        return self.dist_p1.rvs(n)
    
    def log_prob_p0(self, x):
        """Compute log P0(x)"""
        return self.dist_p0.logpdf(x)
    
    def log_prob_p1(self, x):
        """Compute log P1(x)"""
        return self.dist_p1.logpdf(x)
    
    def true_log_ratio(self, x):
        """Compute true log density ratio log(P1(x)/P0(x))"""
        return self.log_prob_p1(x) - self.log_prob_p0(x)
    
    def empirical_kl(self, n_samples=100000):
        """
        Compute empirical KL divergence using Monte Carlo estimation.
        KL(P1||P0) = E_{x~P1}[log P1(x) - log P0(x)]
        """
        samples = self.sample_p1(n_samples)
        log_ratios = self.true_log_ratio(samples)
        return np.mean(log_ratios)


def main():
    """Test the TwoGaussians class"""
    dataset = TwoGaussians(n_samples=1000, n_dims=10, mean_shift=2.0)
    
    print(f"Dimensionality: {dataset.n_dims}")
    print(f"P0 mean: {dataset.mu0[:5]}...")
    print(f"P1 mean: {dataset.mu1[:5]}...")
    print(f"True KL(P1||P0): {dataset.true_kl:.4f}")
    print(f"Empirical KL(P1||P0): {dataset.empirical_kl():.4f}")
    print(f"Training samples: P0={dataset.trn.N_p0}, P1={dataset.trn.N_p1}")
    

if __name__ == "__main__":
    main()
