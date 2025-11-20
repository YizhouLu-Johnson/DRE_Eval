"""
Configuration Generator for BDRE vs TDRE Comparison on 10D Gaussians

This script generates configuration files for training TDRE models
on 10-dimensional Gaussian distributions for comparison with BDRE.
"""

import os
import json
import numpy as np
from copy import deepcopy
from time import gmtime, strftime
from make_configs import make_base_config, save_config, generate_configs_for_gridsearch, update_config


def make_comparison_gaussian_configs():
    """
    Generate configs for 10D Gaussian TDRE experiments.
    These will be compared against BDRE on the same data.
    """
    config = make_base_config()
    
    # Dataset configuration
    config["data"]["dataset_name"] = "gaussians"  # Use existing dataset infrastructure
    config["data"]["n_dims"] = 10
    config["data"]["data_args"] = {
        "n_samples": 10000,
        "n_dims": 10,
        "true_mutual_info": 5.0  # This creates correlation structure
    }
    config["data"]["data_dist_name"] = "gaussian"
    config["data"]["noise_dist_name"] = "gaussian"
    
    # Model architecture - use quadratic for better expressiveness
    config["architecture"]["network_type"] = "quadratic"
    config["architecture"]["quadratic_constraint_type"] = "symmetric_pos_diag"
    config["architecture"]["quadratic_head_use_linear_term"] = True
    
    # Optimization settings
    config["optimisation"]["n_epochs"] = 200
    config["optimisation"]["n_batch"] = 128
    config["optimisation"]["patience"] = 30
    config["optimisation"]["energy_lr"] = 1e-3
    config["optimisation"]["save_every_x_epochs"] = 20
    config["optimisation"]["loss_function"] = "logistic"  # Loss function is in optimisation config
    
    # AIS settings for evaluation
    config["ais"]["ais_n_chains"] = 1000
    config["ais"]["ais_total_n_steps"] = 1000
    
    # Waymark/bridge settings for TDRE
    # Try different numbers of intermediate distributions
    configs_to_generate = []
    
    # Configuration 1: 2 waymarks (single ratio - similar to BDRE)
    config1 = deepcopy(config)
    config1["data"]["linear_combo_alphas"] = [0.0, 1.0]
    config1["data"]["initial_waymark_indices"] = [0, 1]
    configs_to_generate.append(("single_ratio", config1))
    
    # Configuration 2: 3 waymarks (2 ratios - mild telescoping)
    config2 = deepcopy(config)
    config2["data"]["linear_combo_alphas"] = [0.0, 0.5, 1.0]
    config2["data"]["initial_waymark_indices"] = [0, 1, 2]
    configs_to_generate.append(("tre_2_ratios", config2))
    
    # Configuration 3: 5 waymarks (4 ratios - more telescoping)
    config3 = deepcopy(config)
    config3["data"]["linear_combo_alphas"] = [0.0, 0.25, 0.5, 0.75, 1.0]
    config3["data"]["initial_waymark_indices"] = [0, 1, 2, 3, 4]
    configs_to_generate.append(("tre_4_ratios", config3))
    
    # Save configurations
    # Set time_id as a global for save_config to use
    import make_configs
    make_configs.time_id = strftime('%Y%m%d-%H%M', gmtime())
    
    for idx, (name, cfg) in enumerate(configs_to_generate):
        # Use the save_config function from make_configs.py which calls update_config
        # This ensures all required fields are added
        save_config(cfg, "model", idx)
        
        print(f"Saved config {idx}: {name}")
        print(f"  Waymarks: {cfg['data']['initial_waymark_indices']}")
        print(f"  Alphas: {cfg['data']['linear_combo_alphas']}")
    
    print(f"\nGenerated {len(configs_to_generate)} TDRE configurations")
    print("To train TDRE models, run:")
    print("  python build_bridges.py --config_path=gaussians/model/0")
    print("  python build_bridges.py --config_path=gaussians/model/1")
    print("  python build_bridges.py --config_path=gaussians/model/2")


def make_sample_size_sweep_configs():
    """
    Generate multiple configs for different training sample sizes.
    This enables studying sample efficiency.
    """
    # Set time_id as a global for save_config to use
    import make_configs
    make_configs.time_id = strftime('%Y%m%d-%H%M', gmtime())
    
    base_config = make_base_config()
    
    # Dataset configuration
    base_config["data"]["dataset_name"] = "gaussians"  # Use existing dataset infrastructure
    base_config["data"]["n_dims"] = 10
    base_config["data"]["data_dist_name"] = "gaussian"
    base_config["data"]["noise_dist_name"] = "gaussian"
    
    # Model architecture
    base_config["architecture"]["network_type"] = "quadratic"
    base_config["architecture"]["quadratic_constraint_type"] = "symmetric_pos_diag"
    base_config["architecture"]["quadratic_head_use_linear_term"] = True
    
    # Optimization settings
    base_config["optimisation"]["n_epochs"] = 200
    base_config["optimisation"]["patience"] = 30
    base_config["optimisation"]["energy_lr"] = 1e-3
    base_config["optimisation"]["loss_function"] = "logistic"
    
    # Sample sizes to try
    sample_sizes = [50, 100, 200, 400, 800, 1600, 3200]
    
    # TDRE configurations (different waymark settings)
    tre_configs = [
        ("single_ratio", [0.0, 1.0], [0, 1]),
        ("tre_2_ratios", [0.0, 0.5, 1.0], [0, 1, 2]),
        ("tre_4_ratios", [0.0, 0.25, 0.5, 0.75, 1.0], [0, 1, 2, 3, 4]),
    ]
    
    config_idx = 0
    for tre_name, alphas, waymark_idxs in tre_configs:
        for n_samples in sample_sizes:
            config = deepcopy(base_config)
            config["data"]["data_args"] = {
                "n_samples": n_samples,
                "n_dims": 10,
                "true_mutual_info": 5.0  # This creates correlation structure
            }
            config["data"]["linear_combo_alphas"] = alphas
            config["data"]["initial_waymark_indices"] = waymark_idxs
            config["optimisation"]["n_batch"] = min(128, n_samples // 4)
            
            # Use save_config to ensure all required fields are added
            save_config(config, "sweep", config_idx)
            
            print(f"Config {config_idx}: {tre_name}, n_samples={n_samples}")
            config_idx += 1
    
    print(f"\nGenerated {config_idx} sweep configurations")


if __name__ == "__main__":
    print("Generating BDRE vs TDRE comparison configurations...\n")
    make_comparison_gaussian_configs()
    print("\n" + "="*60 + "\n")
    print("Generating sample size sweep configurations...\n")
    make_sample_size_sweep_configs()
    print("\nDone!")

