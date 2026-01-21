from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from experimenter_benchmark.dre_methods import DREMethod, get_dre_method


Array = np.ndarray


@dataclass
class ExperimentConfig:
    # Section: Parameter, Design, and Observation Spaces
    dim: int = 5
    noise_std: float = 0.8
    prior_variance: float = 4.0
    n_rounds: int = 5
    n_design_candidates: int = 8
    design_scale: float = 2.0
    reuse_design_candidates: bool = True
    log_progress: bool = True
    log_eig_per_candidate: bool = True
    dre_method: str = "tdre"
    tdre_num_samples: int = 1500
    tdre_eval_samples: int = 800
    tdre_n_waymarks: int = 12
    tdre_n_epochs: int = 500
    tdre_patience: int = 100
    tdre_batch_size: int = 128
    tdre_energy_lr: float = 5e-4
    tdre_val_interval: int = 10
    tdre_loss_function: str = "logistic"
    tdre_optimizer: str = "adam"
    tdre_loss_decay_factor: float = 1.0
    tdre_network_type: str = "quadratic"
    tdre_quadratic_constraint_type: str = "semi_pos_def"
    tdre_quadratic_use_linear_term: bool = True
    tdre_mlp_hidden_size: int = 256
    tdre_mlp_n_blocks: int = 2
    tdre_eig_repeats: int = 5
    tdre_waymark_mechanism: str = "linear_combinations"
    tdre_shuffle_waymarks: bool = False
    tdre_save_root: str = "results/seq_design"
    tdre_script_path: Optional[str] = None
    tdre_config_dir: str = "seq_design"
    rng_seed: int = 0
    prior_mean: Optional[Array] = None
    prior_cov: Optional[Array] = None


@dataclass
class ExperimentResult:
    designs: List[Array] = field(default_factory=list)
    observations: List[float] = field(default_factory=list)
    eig_estimates: List[float] = field(default_factory=list)
    posterior_means: List[Array] = field(default_factory=list)
    posterior_covs: List[Array] = field(default_factory=list)
    candidate_eigs: List[Array] = field(default_factory=list)
    theta_star: Optional[Array] = None


class LinearGaussianSimulator:
    # Section: Forward Model, Simulator, and Environment
    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def sample(self, theta: Array, design: Array, rng: np.random.Generator) -> Array:
        mean = theta @ design
        noise = rng.normal(0.0, self.noise_std, size=mean.shape)
        return mean + noise


class GroundTruthEnvironment:
    # Section: Prior and Ground Truth
    def __init__(self, theta_star: Array, noise_std: float):
        self.theta_star = theta_star
        self.noise_std = noise_std

    def observe(self, design: Array, rng: np.random.Generator) -> float:
        mean = float(self.theta_star @ design)
        noise = float(rng.normal(0.0, self.noise_std))
        return mean + noise


class GaussianPosterior:
    # Section: Prior and Ground Truth
    def __init__(self, mean: Array, cov: Array):
        self.mean = mean
        self.cov = cov

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def update(self, design: Array, observation: float, noise_std: float) -> "GaussianPosterior":
        # Section: Posterior Update (Gaussian Benchmark)
        sigma2 = noise_std ** 2
        precision = np.linalg.inv(self.cov)
        precision_update = precision + (1.0 / sigma2) * np.outer(design, design)
        cov_new = np.linalg.inv(precision_update)
        mean_new = cov_new @ (precision @ self.mean + (1.0 / sigma2) * design * observation)
        return GaussianPosterior(mean_new, cov_new)


class GaussianDesignSampler:
    # Section: Parameter, Design, and Observation Spaces
    def __init__(self, dim: int, scale: float):
        self.dim = dim
        self.scale = scale

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        return rng.normal(0.0, self.scale, size=(n, self.dim))


def _joint_and_product_covariances(
    prior: GaussianPosterior,
    design: Array,
    noise_std: float,
) -> Tuple[Array, Array, Array]:
    # Section: Joint and Product Distributions
    mean_theta = prior.mean
    mean_y = float(mean_theta @ design)
    joint_mean = np.concatenate([mean_theta, np.array([mean_y])])

    sigma = prior.cov
    cross = sigma @ design
    var_y = float(design @ sigma @ design + noise_std ** 2)

    joint_cov = np.block([
        [sigma, cross[:, None]],
        [cross[None, :], np.array([[var_y]])],
    ])
    product_cov = np.block([
        [sigma, np.zeros((sigma.shape[0], 1))],
        [np.zeros((1, sigma.shape[0])), np.array([[var_y]])],
    ])

    return joint_mean, joint_cov, product_cov


def estimate_eig(
    prior: GaussianPosterior,
    simulator: LinearGaussianSimulator,
    design: Array,
    dre_method: DREMethod,
    rng: np.random.Generator,
) -> float:
    # Section: Expected Information Gain (EIG)
    joint_mean, joint_cov, product_cov = _joint_and_product_covariances(prior, design, simulator.noise_std)
    return dre_method.estimate_eig(joint_mean, joint_cov, product_cov, rng)


def select_design(
    prior: GaussianPosterior,
    simulator: LinearGaussianSimulator,
    candidates: Array,
    dre_method: DREMethod,
    rng: np.random.Generator,
    config: ExperimentConfig,
    round_idx: int,
) -> Tuple[Array, Array]:
    eig_values = []
    total = candidates.shape[0]
    for idx, design in enumerate(candidates):
        if config.log_progress:
            print(f"[round {round_idx + 1}] training design {idx + 1}/{total}")
        eig_runs = []
        for repeat_idx in range(config.tdre_eig_repeats):
            if config.log_progress and config.tdre_eig_repeats > 1:
                print(f"[round {round_idx + 1}] design {idx + 1} repeat {repeat_idx + 1}/{config.tdre_eig_repeats}")
            eig_runs.append(estimate_eig(prior, simulator, design, dre_method, rng))
        eig = float(np.mean(eig_runs))
        if config.log_eig_per_candidate:
            print(f"[round {round_idx + 1}] design {idx + 1} EIG (avg): {eig:.6f}")
        eig_values.append(eig)
    eig_values = np.array(eig_values)
    best_idx = int(np.argmax(eig_values))
    return candidates[best_idx], eig_values


def _make_prior(config: ExperimentConfig) -> GaussianPosterior:
    mean = np.zeros(config.dim) if config.prior_mean is None else np.array(config.prior_mean, dtype=float)
    if config.prior_cov is None:
        cov = np.eye(config.dim) * config.prior_variance
    else:
        cov = np.array(config.prior_cov, dtype=float)
    return GaussianPosterior(mean, cov)


def build_fixed_candidates(config: ExperimentConfig) -> Array:
    rng = np.random.default_rng(config.rng_seed)
    sampler = GaussianDesignSampler(config.dim, config.design_scale)
    return sampler.sample(config.n_design_candidates, rng)


def _squared_distance(a: Array, b: Array) -> float:
    diff = a - b
    return float(diff @ diff)


def run_sequential_design(
    config: ExperimentConfig,
    fixed_candidates: Optional[Array] = None,
) -> ExperimentResult:
    # Section: Sequential Experiment Loop
    rng = np.random.default_rng(config.rng_seed)
    prior = _make_prior(config)
    simulator = LinearGaussianSimulator(config.noise_std)
    dre_method = get_dre_method(config)

    theta_star = prior.sample(1, rng).reshape(-1)
    environment = GroundTruthEnvironment(theta_star, config.noise_std)
    design_sampler = GaussianDesignSampler(config.dim, config.design_scale)
    if config.reuse_design_candidates and fixed_candidates is None:
        fixed_candidates = design_sampler.sample(config.n_design_candidates, rng)

    result = ExperimentResult(theta_star=theta_star)
    result.posterior_means.append(prior.mean.copy())
    result.posterior_covs.append(prior.cov.copy())

    for round_idx in range(config.n_rounds):
        if fixed_candidates is None:
            candidates = design_sampler.sample(config.n_design_candidates, rng)
        else:
            candidates = fixed_candidates
        design, eig_values = select_design(
            prior,
            simulator,
            candidates,
            dre_method,
            rng,
            config,
            round_idx,
        )

        observation = environment.observe(design, rng)

        # ADD THESE DEBUG LINES:
        print(f"\n[DEBUG round {round_idx + 1}]")
        print(f"  Selected design: {design}")
        print(f"  θ*: {theta_star}")
        print(f"  Expected obs (θ*^T ξ): {float(theta_star @ design):.6f}")
        print(f"  Actual obs: {observation:.6f}")
        print(f"  Prior mean before update: {prior.mean}")
        print(f"  Prior cov trace before: {np.trace(prior.cov):.6f}")

        prior = prior.update(design, observation, config.noise_std)

        # ADD MORE DEBUG:
        print(f"  Posterior mean after update: {prior.mean}")
        print(f"  Posterior cov trace after: {np.trace(prior.cov):.6f}")
        print(f"  Sq dist before: {_squared_distance(result.posterior_means[-1], theta_star):.6f}")
        print(f"  Sq dist after: {_squared_distance(prior.mean, theta_star):.6f}")

        result.designs.append(design)
        result.observations.append(observation)
        result.eig_estimates.append(float(eig_values.max()))
        result.candidate_eigs.append(eig_values)
        result.posterior_means.append(prior.mean.copy())
        result.posterior_covs.append(prior.cov.copy())
        if config.log_progress:
            sq_dist = _squared_distance(prior.mean, theta_star)
            print(f"[round {round_idx + 1}] selected EIG: {float(eig_values.max()):.6f}")
            print(f"[round {round_idx + 1}] obs: {observation:.6f} | sq dist: {sq_dist:.6f}")

    return result


def run_random_design(
    config: ExperimentConfig,
    theta_star: Array,
    fixed_candidates: Optional[Array] = None,
) -> ExperimentResult:
    rng = np.random.default_rng(config.rng_seed + 1)
    prior = _make_prior(config)
    simulator = LinearGaussianSimulator(config.noise_std)
    environment = GroundTruthEnvironment(theta_star, config.noise_std)
    design_sampler = GaussianDesignSampler(config.dim, config.design_scale)

    result = ExperimentResult(theta_star=theta_star)
    result.posterior_means.append(prior.mean.copy())
    result.posterior_covs.append(prior.cov.copy())

    for round_idx in range(config.n_rounds):
        # Randomly select ONE design (fair comparison with TDRE)
        if fixed_candidates is None:
            candidates = design_sampler.sample(config.n_design_candidates, rng)
        else:
            candidates = fixed_candidates
        
        # Randomly choose one design
        chosen_idx = int(rng.integers(0, candidates.shape[0]))
        design = candidates[chosen_idx]
        
        # Take ONE observation (same as TDRE)
        observation = environment.observe(design, rng)
        
        # Update posterior once
        prior = prior.update(design, observation, config.noise_std)

        result.designs.append(design)
        result.observations.append(observation)
        result.eig_estimates.append(float("nan"))
        result.candidate_eigs.append(np.full(candidates.shape[0], np.nan))
        result.posterior_means.append(prior.mean.copy())
        result.posterior_covs.append(prior.cov.copy())
        
        if config.log_progress:
            sq_dist = _squared_distance(prior.mean, theta_star)
            print(f"[random round {round_idx + 1}] design {chosen_idx + 1}/{candidates.shape[0]} | obs: {observation:.6f} | sq dist: {sq_dist:.6f}")

    return result
