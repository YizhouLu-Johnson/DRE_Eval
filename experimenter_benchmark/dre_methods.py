from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from experimenter_benchmark.tdre_config import (
    TDREConfigInputs,
    append_tdre_script_line,
    build_tdre_config,
    write_tdre_config,
)
from utils.experiment_utils import load_data_providers_and_update_conf
from build_bridges import build_graph as build_tdre_graph
from build_bridges import get_feed_dict as tre_get_feed_dict
from build_bridges import make_logger as tre_make_logger
from build_bridges import make_savers as tre_make_savers
from build_bridges import train as tre_train


class DREMethod(Protocol):
    def estimate_eig(
        self,
        joint_mean: np.ndarray,
        joint_cov: np.ndarray,
        product_cov: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        ...


@dataclass
class TDREMethod:
    config: object

    def estimate_eig(
        self,
        joint_mean: np.ndarray,
        joint_cov: np.ndarray,
        product_cov: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        tdre_inputs = TDREConfigInputs(
            dim=int(joint_cov.shape[0]),
            num_samples=int(self.config.tdre_num_samples),
            n_waymarks=int(self.config.tdre_n_waymarks),
            n_epochs=int(self.config.tdre_n_epochs),
            patience=int(self.config.tdre_patience),
            batch_size=int(self.config.tdre_batch_size),
            energy_lr=float(self.config.tdre_energy_lr),
            loss_function=str(self.config.tdre_loss_function),
            optimizer=str(self.config.tdre_optimizer),
            loss_decay_factor=float(self.config.tdre_loss_decay_factor),
            network_type=str(self.config.tdre_network_type),
            quadratic_constraint_type=str(self.config.tdre_quadratic_constraint_type),
            quadratic_use_linear_term=bool(self.config.tdre_quadratic_use_linear_term),
            mlp_hidden_size=int(self.config.tdre_mlp_hidden_size),
            mlp_n_blocks=int(self.config.tdre_mlp_n_blocks),
            waymark_mechanism=str(self.config.tdre_waymark_mechanism),
            shuffle_waymarks=bool(self.config.tdre_shuffle_waymarks),
            save_root=None,
        )

        seed = int(rng.integers(0, 2**31 - 1))
        config_id = f"seq_{seed}"
        config_dir_name = getattr(self.config, "tdre_config_dir", "seq_design")
        save_dir_root = str(self.config.tdre_save_root)
        tdre_config, tdre_config_flat = build_tdre_config(
            tdre_inputs,
            joint_cov=joint_cov,
            product_cov=product_cov,
            rng_seed=seed,
            config_dir_name=config_dir_name,
            save_dir_root=save_dir_root,
            time_id=config_id,
            config_id=config_id,
        )
        config_path = Path("configs") / config_dir_name / "model" / f"{config_id}.json"
        write_tdre_config(tdre_config, config_path)
        if self.config.tdre_script_path:
            config_ref = f"{config_dir_name}/model/{config_id}"
            append_tdre_script_line(Path(self.config.tdre_script_path), config_path, config_ref)

        tre_make_logger()
        tf.reset_default_graph()
        graph = build_tdre_graph(tdre_config_flat)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_dp, val_dp = load_data_providers_and_update_conf(tdre_config_flat)
        saver1, saver2 = tre_make_savers(tdre_config_flat)
        tre_train(graph, sess, train_dp, val_dp, saver1, saver2, tdre_config_flat)

        eval_count = min(int(self.config.tdre_eval_samples), val_dp.data.shape[0])
        eval_batch = val_dp.data[:eval_count].astype(np.float32)
        eval_feed = tre_get_feed_dict(graph, sess, val_dp, eval_batch, tdre_config_flat, train=False)
        neg_energies = sess.run(graph.neg_energies_of_data, feed_dict=eval_feed)
        sess.close()

        log_ratio = np.sum(neg_energies, axis=1)
        return float(np.mean(log_ratio))


def get_dre_method(config: object) -> DREMethod:
    method = getattr(config, "dre_method", "tdre")
    if method == "tdre":
        return TDREMethod(config)
    raise ValueError(f"Unknown DRE method: {method}")
