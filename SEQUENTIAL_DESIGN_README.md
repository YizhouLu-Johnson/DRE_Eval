# Sequential Bayesian Experimental Design with TDRE

## Overview

This implementation provides a modular framework for **Sequential Bayesian Experimental Design** using **Density Ratio Estimation (DRE)** methods, specifically **Targeted Density Ratio Estimation (TDRE)**. The framework follows the mathematical formulation described in the project writeup and is designed to be extensible to other DRE methods (DV, NWJ, classifier-based DRE) and different simulators/distributions.

## Mathematical Background

### Problem Setup

- **Parameter space**: θ ∈ ℝ^d (default d=5)
- **Design space**: ξ ∈ ℝ^5
- **Observation space**: y ∈ ℝ

### Forward Model

The data-generating process is linear-Gaussian:

```
y = θ^T ξ + ε,  where ε ~ N(0, σ²)
```

This induces the likelihood:

```
p(y | ξ, θ) = N(y; θ^T ξ, σ²)
```

### Sequential Experiment Loop

At each round t = 1, 2, ..., T:

1. **Select design**: Choose ξ_t to maximize Expected Information Gain (EIG)
2. **Execute experiment**: Observe y_t ~ μ(· | ξ_t) from the environment
3. **Update belief**: Compute posterior p_t(θ) given data

### Expected Information Gain (EIG)

EIG is defined as the KL divergence between joint and product distributions:

```
EIG(ξ) = KL(p_0(θ,y|ξ) || p_1(θ,y|ξ))
```

where:
- **Joint distribution**: p_0(θ,y|ξ) = p_{t-1}(θ) · k(y|ξ,θ)
- **Product distribution**: p_1(θ,y|ξ) = p_{t-1}(θ) · c_{t-1}(y|ξ)
- **Posterior predictive**: c_{t-1}(y|ξ) = ∫ p_{t-1}(θ) k(y|ξ,θ) dθ

### Density Ratio Estimation

The density ratio is:

```
r(θ,y; ξ) = p_0(θ,y|ξ) / p_1(θ,y|ξ) = k(y|ξ,θ) / c_{t-1}(y|ξ)
```

Then:

```
EIG(ξ) = E_{p_0}[log r(θ,y; ξ)]
```

### TDRE (Targeted Density Ratio Estimation)

TDRE estimates the density ratio via **moment matching** (not ELBO):

```
min_φ Σ_j (E_{p_1}[r_φ(x) f_j(x)] - E_{p_0}[f_j(x)])² + λ R(φ)
```

Once r_φ is obtained, EIG is estimated as:

```
EIG(ξ) ≈ (1/N) Σ_i log r_φ(x_i),  where x_i ~ p_0
```

### Posterior Update (Gaussian Case)

With Gaussian prior and likelihood, the posterior remains Gaussian:

```
Σ_t = (Σ_{t-1}^{-1} + (1/σ²) ξ_t ξ_t^T)^{-1}
m_t = Σ_t (Σ_{t-1}^{-1} m_{t-1} + (1/σ²) ξ_t y_t)
```

## Implementation Structure

### Core Components

The implementation is organized into modular components that correspond to sections of the mathematical formulation:

#### 1. **Parameter, Design, and Observation Spaces** (`GaussianDesignSampler`)

```python
class GaussianDesignSampler:
    """Samples candidate designs from Gaussian distribution."""
    def sample(self, n: int, rng) -> Array:
        return rng.normal(0.0, self.scale, size=(n, self.dim))
```

#### 2. **Prior and Ground Truth** (`GaussianPosterior`, `GroundTruthEnvironment`)

```python
class GaussianPosterior:
    """Represents Gaussian belief over parameters."""
    def __init__(self, mean: Array, cov: Array)
    def sample(self, n: int, rng) -> Array
    def update(self, design, observation, noise_std) -> GaussianPosterior
```

```python
class GroundTruthEnvironment:
    """Simulates real-world observations from fixed θ*."""
    def __init__(self, theta_star: Array, noise_std: float)
    def observe(self, design: Array, rng) -> float
```

#### 3. **Forward Model and Simulator** (`LinearGaussianSimulator`)

```python
class LinearGaussianSimulator:
    """Implements k(y|ξ,θ) = N(y; θ^T ξ, σ²)."""
    def sample(self, theta: Array, design: Array, rng) -> Array
```

#### 4. **Joint and Product Distributions** (`_joint_and_product_covariances`)

```python
def _joint_and_product_covariances(prior, design, noise_std):
    """
    Computes covariances for:
    - Joint: p_0(θ,y|ξ) with correlation between θ and y
    - Product: p_1(θ,y|ξ) with independent θ and y
    """
    # Returns: joint_mean, joint_cov, product_cov
```

#### 5. **Density Ratio Estimation - TDRE** (`TDREMethod`)

```python
class TDREMethod:
    """TDRE implementation using build_bridges.py in this repo."""
    
    def estimate_eig(self, joint_mean, joint_cov, product_cov, rng) -> float:
        """
        Trains TDRE model and estimates EIG.
        
        Process:
        1. Build a config matching make_gaussians_10d_configs.py
        2. Instantiate GAUSSIANS via local data providers
        3. Build TensorFlow graph with quadratic energy network
        4. Train with build_bridges.py (train function)
        5. Evaluate log-ratio on validation samples
        6. Return mean log-ratio as EIG estimate
        """
```

**Key TDRE Configuration Parameters:**
- `tdre_num_samples`: Training samples (default: 256)
- `tdre_n_waymarks`: Number of intermediate distributions (default: 8)
- `tdre_n_epochs`: Training epochs (default: 200)
- `tdre_patience`: Early stopping patience (default: 50)
- `tdre_network_type`: "quadratic" (uses quadratic heads)
- `tdre_loss_function`: "logistic" (logistic loss)

#### 6. **Expected Information Gain** (`estimate_eig`, `select_design`)

```python
def estimate_eig(prior, simulator, design, config, rng) -> float:
    """Estimates EIG for a single design using TDRE."""
    joint_mean, joint_cov, product_cov = _joint_and_product_covariances(...)
    dre_method = get_dre_method(config)
    return dre_method.estimate_eig(joint_mean, joint_cov, product_cov, rng)

def select_design(prior, simulator, candidates, config, rng):
    """Selects design that maximizes EIG among candidates."""
    eig_values = [estimate_eig(..., design, ...) for design in candidates]
    return candidates[argmax(eig_values)], eig_values
```

#### 7. **Sequential Experiment Loop** (`run_sequential_design`)

```python
def run_sequential_design(config: ExperimentConfig) -> ExperimentResult:
    """
    Main loop implementing sequential Bayesian experimental design.
    
    For t = 1, ..., T:
        1. Sample design candidates
        2. Estimate EIG for each candidate using TDRE
        3. Select design with maximum EIG
        4. Observe data from environment
        5. Update Gaussian posterior
        6. Store results
    """
```

## File Structure

```
experimenter-benchmark/
├── experimenter_benchmark/
│   ├── __init__.py
│   ├── dre_methods.py                # DRE method interface + registry
│   ├── sequential_design.py          # Main experiment loop (method-agnostic)
│   └── tdre_config.py                # TDRE config builder (make_gaussians_10d_configs.py style)
├── run_experimenter_benchmark.py      # Command-line interface
└── SEQUENTIAL_DESIGN_README.md        # This file
```

## Usage

### Basic Usage

```python
from experimenter_benchmark.sequential_design import (
    ExperimentConfig,
    run_sequential_design,
)

# Configure experiment
config = ExperimentConfig(
    dim=5,                      # Parameter dimension
    noise_std=1.0,              # Observation noise
    n_rounds=10,                # Number of experimental rounds
    n_design_candidates=64,     # Designs to evaluate per round
    design_scale=1.0,           # Scale of design sampling
    tdre_num_samples=256,       # TDRE training samples
    tdre_n_waymarks=8,          # TDRE waymarks
    tdre_n_epochs=200,          # TDRE training epochs
    rng_seed=42,                # Random seed
)

# Run sequential design
result = run_sequential_design(config)

# Access results
print(f"True parameter: {result.theta_star}")
print(f"Final estimate: {result.posterior_means[-1]}")
print(f"Designs: {result.designs}")
print(f"Observations: {result.observations}")
print(f"EIG estimates: {result.eig_estimates}")
```

### Command-Line Interface

```bash
python run_experimenter_benchmark.py \
    --dim 5 \
    --noise_std 1.0 \
    --rounds 10 \
    --design_candidates 64 \
    --tdre_samples 256 \
    --tdre_waymarks 8 \
    --tdre_epochs 200 \
    --seed 42 \
    --save_path results.npz
```

Complete run with fixed candidates and random baseline:

```bash
python run_experimenter_benchmark.py \
    --dre_method tdre \
    --compare_random
```

This prints the squared distance between the final posterior mean and the true
θ* for both TDRE (max‑EIG) and the random baseline.

### TDRE Step-by-Step (sequential design loop)

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dre
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/Experimenter-benchmark
python run_experimenter_benchmark.py --dre_method tdre
```

This runs the sequential design loop and trains TDRE internally each round using
`experimenter_benchmark/tdre_config.py` and `build_bridges.py`.
It prints per-candidate EIGs and round summaries by default. Use `--quiet` to suppress.

During each EIG evaluation, a TDRE config JSON is written to:

```
configs/seq_design/model/seq_<seed>.json
```

That JSON is in the same nested format as `configs/gaussians_10d/model/*.json`,
so it can be passed directly to:

```bash
python build_bridges.py --config_path=seq_design/model/seq_<seed>
```

### Random Baseline

Use `--compare_random` to run an additional baseline that chooses a random
design each round (no EIG optimization). The final squared distance to θ* is
printed for both runs so you can compare performance.

### TDRE Step-by-Step (standalone config generation + build_bridges)

Generate configs in `configs/gaussians_10d/model/`:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dre
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/Experimenter-benchmark
python make_gaussians_10d_configs.py
```

Train TDRE for all generated configs:

```bash
for cfg in configs/gaussians_10d/model/*.json; do
  python build_bridges.py --config_path="$cfg"
done
```

Override defaults if needed:

```bash
python make_gaussians_10d_configs.py \
  --target_kls 10 15 20 \
  --sample_sizes 50 100 300 \
  --n_trials 10
```

Disable candidate reuse or progress logging:

```bash
python run_experimenter_benchmark.py --no_reuse_candidates --quiet
```

### Testing

Run the comprehensive test suite:

```bash
python test_sequential_design.py
```

This will:
1. Run 3 rounds of sequential design
2. Verify error decreases
3. Check EIG estimates are positive
4. Validate posterior updates
5. Generate visualization plots

Run basic component tests:

```bash
python test_simple.py
```

## Configuration Parameters

### Experiment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 5 | Dimension of parameter space θ |
| `noise_std` | float | 1.0 | Standard deviation of observation noise σ |
| `n_rounds` | int | 10 | Number of sequential experimental rounds |
| `n_design_candidates` | int | 64 | Number of candidate designs to evaluate per round |
| `design_scale` | float | 1.0 | Scale parameter for design sampling |
| `dre_method` | str | "tdre" | DRE method selector |
| `rng_seed` | int | 0 | Random seed for reproducibility |

### TDRE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tdre_num_samples` | int | 300 | Number of training samples for TDRE |
| `tdre_eval_samples` | int | 300 | Number of evaluation samples for EIG |
| `tdre_n_waymarks` | int | 12 | Number of waymark distributions |
| `tdre_n_epochs` | int | 400 | Maximum training epochs |
| `tdre_patience` | int | 100 | Early stopping patience |
| `tdre_batch_size` | int | 128 | Training batch size |
| `tdre_energy_lr` | float | 5e-4 | Learning rate for energy network |
| `tdre_val_interval` | int | 10 | Validation check interval |
| `tdre_loss_function` | str | "logistic" | Loss function (logistic/nwj/lsq) |
| `tdre_optimizer` | str | "adam" | Optimizer (adam/rmsprop/momentum) |
| `tdre_network_type` | str | "quadratic" | Network architecture |
| `tdre_quadratic_constraint_type` | str | "semi_pos_def" | Constraint on quadratic form |
| `tdre_save_root` | str | "results/seq_design" | Output directory for TDRE runs |
| `tdre_script_path` | str | None | Optional shell script to append `build_bridges.py` calls |
| `tdre_config_dir` | str | "seq_design" | Config subdirectory under `configs/` for TDRE runs |
| `reuse_design_candidates` | bool | True | Reuse the same candidate designs each round |
| `log_progress` | bool | True | Print round-by-round progress |
| `log_eig_per_candidate` | bool | True | Print EIG for each candidate design |

## Results Format

The `ExperimentResult` object contains:

```python
@dataclass
class ExperimentResult:
    designs: List[Array]              # Selected designs [ξ_1, ..., ξ_T]
    observations: List[float]         # Observed data [y_1, ..., y_T]
    eig_estimates: List[float]        # EIG of selected designs
    posterior_means: List[Array]      # [m_0, m_1, ..., m_T]
    posterior_covs: List[Array]       # [Σ_0, Σ_1, ..., Σ_T]
    candidate_eigs: List[Array]       # EIG for all candidates per round
    theta_star: Array                 # Ground truth parameter
```

Results can be saved to `.npz` format:

```python
np.savez(
    "results.npz",
    designs=np.stack(result.designs),
    observations=np.array(result.observations),
    eig_estimates=np.array(result.eig_estimates),
    posterior_means=np.stack(result.posterior_means),
    posterior_covs=np.stack(result.posterior_covs),
    candidate_eigs=np.stack(result.candidate_eigs),
    theta_star=result.theta_star,
)
```

## Visualization

The test script generates a comprehensive visualization with 4 subplots:

1. **Parameter Estimation Error**: ||θ̂_t - θ*||₂ vs round (should decrease)
2. **Expected Information Gain**: EIG per round (should be positive)
3. **Posterior Uncertainty**: Tr(Σ_t) vs round (should decrease)
4. **Component-wise Estimates**: Individual θ[i] trajectories

## Validation Criteria

A successful run should exhibit:

1. ✅ **Decreasing estimation error**: ||θ̂_T - θ*|| < ||θ̂_0 - θ*||
2. ✅ **Positive EIG estimates**: All EIG values > 0
3. ✅ **Decreasing uncertainty**: Tr(Σ_T) < Tr(Σ_0)
4. ✅ **TDRE convergence**: Training completes without errors

## Extensibility

### Adding New DRE Methods

To add a new DRE method (e.g., DV, NWJ, classifier-based):

1. Create a new class that implements the `DREMethod` protocol (see `experimenter_benchmark/dre_methods.py`):

```python
class NewDREMethod:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def estimate_eig(self, joint_mean, joint_cov, product_cov, rng) -> float:
        # Implement your DRE method here
        # Return EIG estimate
        pass
```

2. Register the method in `get_dre_method`:

```python
def get_dre_method(config: ExperimentConfig) -> DREMethod:
    if config.dre_method == "tdre":
        return TDREMethod(config)
    if config.dre_method == "new_method":
        return NewDREMethod(config)
    raise ValueError("Unknown DRE method")
```

### Adding New Simulators

To use a different forward model:

1. Create a new simulator class:

```python
class NewSimulator:
    def sample(self, theta: Array, design: Array, rng) -> Array:
        # Implement your forward model
        pass
```

2. Modify `_joint_and_product_covariances` or create a new function for non-Gaussian cases

3. Update `run_sequential_design` to use the new simulator

### Adding Non-Gaussian Posteriors

For non-Gaussian posteriors, replace `GaussianPosterior` with:

```python
class ParticlePosterior:
    def __init__(self, particles: Array, weights: Array):
        self.particles = particles
        self.weights = weights
    
    def sample(self, n: int, rng) -> Array:
        # Resample from particles
        pass
    
    def update(self, design, observation, simulator):
        # Particle filter update
        pass
```

## Implementation Notes

### TDRE Integration

The implementation integrates with local TDRE code by:

1. **Config builder**: `tdre_config.build_tdre_config` produces a nested JSON config compatible with `build_bridges.py`
2. **Data providers**: `load_data_providers_and_update_conf` builds GAUSSIANS datasets
3. **Graph build**: `build_bridges.py` creates the TDRE model
4. **Training loop**: `build_bridges.py` runs the optimization
5. **Evaluation**: TDRE computes mean log‑ratio after training
