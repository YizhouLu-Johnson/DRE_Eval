from __future__ import annotations

from argparse import ArgumentParser
import os
import sys
from time import strftime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experimenter_benchmark.sequential_design import (
    ExperimentConfig,
    build_fixed_candidates,
    run_random_design,
    run_sequential_design,
)


def _save_results(save_path: str, result) -> None:
    designs = np.stack(result.designs, axis=0)
    observations = np.array(result.observations)
    eig_estimates = np.array(result.eig_estimates)
    posterior_means = np.stack(result.posterior_means, axis=0)
    posterior_covs = np.stack(result.posterior_covs, axis=0)
    candidate_eigs = np.stack(result.candidate_eigs, axis=0)

    np.savez(
        save_path,
        designs=designs,
        observations=observations,
        eig_estimates=eig_estimates,
        posterior_means=posterior_means,
        posterior_covs=posterior_covs,
        candidate_eigs=candidate_eigs,
        theta_star=result.theta_star,
    )


def _squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff @ diff)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _write_distance_csv(path: str, label: str, distances: np.ndarray) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        if handle.tell() == 0:
            handle.write("round,method,sq_distance\n")
        for idx, dist in enumerate(distances, start=1):
            handle.write(f"{idx},{label},{dist:.6f}\n")


def _plot_distances(path: str, tdre_distances: np.ndarray, random_distances: np.ndarray | None) -> None:
    rounds = np.arange(1, len(tdre_distances) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(rounds, tdre_distances, label="tdre", marker="o")
    if random_distances is not None:
        plt.plot(rounds, random_distances, label="random", marker="o")
    plt.xlabel("Round")
    plt.ylabel("Squared distance")
    plt.title("Squared distance to θ* over rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = ArgumentParser(description="Sequential design benchmark with Gaussian model + TDRE.")
    base_config = ExperimentConfig()
    parser.add_argument("--dim", type=int, default=base_config.dim)
    parser.add_argument("--noise_std", type=float, default=base_config.noise_std)
    parser.add_argument("--prior_variance", type=float, default=base_config.prior_variance)
    parser.add_argument("--rounds", type=int, default=base_config.n_rounds)
    parser.add_argument("--design_candidates", type=int, default=base_config.n_design_candidates)
    parser.add_argument("--design_scale", type=float, default=base_config.design_scale)
    parser.add_argument("--dre_method", type=str, default=base_config.dre_method)
    parser.add_argument("--tdre_samples", type=int, default=base_config.tdre_num_samples)
    parser.add_argument("--tdre_eval_samples", type=int, default=base_config.tdre_eval_samples)
    parser.add_argument("--tdre_waymarks", type=int, default=base_config.tdre_n_waymarks)
    parser.add_argument("--tdre_epochs", type=int, default=base_config.tdre_n_epochs)
    parser.add_argument("--tdre_patience", type=int, default=base_config.tdre_patience)
    parser.add_argument("--tdre_batch_size", type=int, default=base_config.tdre_batch_size)
    parser.add_argument("--tdre_energy_lr", type=float, default=base_config.tdre_energy_lr)
    parser.add_argument("--tdre_val_interval", type=int, default=base_config.tdre_val_interval)
    parser.add_argument("--tdre_loss_decay", type=float, default=base_config.tdre_loss_decay_factor)
    parser.add_argument("--tdre_save_root", type=str, default=base_config.tdre_save_root)
    parser.add_argument("--tdre_script_path", type=str, default="")
    parser.add_argument("--tdre_config_dir", type=str, default=base_config.tdre_config_dir)
    parser.add_argument("--tdre_eig_repeats", type=int, default=base_config.tdre_eig_repeats)
    parser.add_argument("--seed", type=int, default=base_config.rng_seed)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--compare_random", action="store_true")
    parser.add_argument("--no_reuse_candidates", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results_dir = "exp_design_results"
    os.makedirs(results_dir, exist_ok=True)
    run_id = strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(results_dir, f"run_{run_id}.txt")
    distances_path = os.path.join(results_dir, f"distances_{run_id}.csv")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        stdout_tee = _Tee(sys.stdout, log_handle)
        stderr_tee = _Tee(sys.stderr, log_handle)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        try:
            config = ExperimentConfig(
                dim=args.dim,
                noise_std=args.noise_std,
                prior_variance=args.prior_variance,
                n_rounds=args.rounds,
                n_design_candidates=args.design_candidates,
                design_scale=args.design_scale,
                dre_method=args.dre_method,
                tdre_num_samples=args.tdre_samples,
                tdre_eval_samples=args.tdre_eval_samples,
                tdre_n_waymarks=args.tdre_waymarks,
                tdre_n_epochs=args.tdre_epochs,
                tdre_patience=args.tdre_patience,
                tdre_batch_size=args.tdre_batch_size,
                tdre_energy_lr=args.tdre_energy_lr,
                tdre_val_interval=args.tdre_val_interval,
                tdre_loss_decay_factor=args.tdre_loss_decay,
                tdre_save_root=args.tdre_save_root,
                tdre_script_path=args.tdre_script_path or None,
                tdre_config_dir=args.tdre_config_dir,
                tdre_eig_repeats=args.tdre_eig_repeats,
                rng_seed=args.seed,
                reuse_design_candidates=not args.no_reuse_candidates,
                log_progress=not args.quiet,
                log_eig_per_candidate=not args.quiet,
            )

            fixed_candidates = None
            if config.reuse_design_candidates:
                fixed_candidates = build_fixed_candidates(config)

            result = run_sequential_design(config, fixed_candidates=fixed_candidates)
            sq_dist = _squared_distance(result.posterior_means[-1], result.theta_star)
            per_round_sq = np.array(
                [_squared_distance(mean, result.theta_star) for mean in result.posterior_means[1:]]
            )

            print("Sequential design benchmark complete.")
            print(f"Theta star: {result.theta_star}")
            print(f"Final posterior mean: {result.posterior_means[-1]}")
            print(f"Final posterior diag: {np.diag(result.posterior_covs[-1])}")
            print(f"Final squared distance: {sq_dist:.6f}")

            _write_distance_csv(distances_path, "tdre", per_round_sq)
            random_per_round_sq = None

            if args.save_path:
                _save_results(args.save_path, result)
                print(f"Saved results to {args.save_path}")

            if args.compare_random:
                print("Running random-design baseline...")
                random_result = run_random_design(
                    config,
                    theta_star=result.theta_star,
                    fixed_candidates=fixed_candidates,
                )
                random_sq_dist = _squared_distance(random_result.posterior_means[-1], random_result.theta_star)
                random_per_round_sq = np.array(
                    [_squared_distance(mean, random_result.theta_star) for mean in random_result.posterior_means[1:]]
                )
                print("Random-design baseline complete.")
                print(f"Random final posterior mean: {random_result.posterior_means[-1]}")
                print(f"Random final squared distance: {random_sq_dist:.6f}")
                _write_distance_csv(distances_path, "random", random_per_round_sq)

            plot_path = os.path.join(results_dir, f"distance_plot_{run_id}.png")
            _plot_distances(plot_path, per_round_sq, random_per_round_sq)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Saved log to {log_path}")
    print(f"Saved per-round distances to {distances_path}")


if __name__ == "__main__":
    main()
