"""
Generate improved TDRE configs with better architecture and training settings.
"""

import os
import json
from copy import deepcopy
from time import gmtime, strftime
from make_configs import make_base_config, save_config


def make_improved_tdre_configs():
    """
    Generate improved TDRE configs with:
    - More waymarks (better approximation)
    - Larger network (more capacity)
    - Longer training (better convergence)
    """
    
    # Set time_id for save_config
    import make_configs
    make_configs.time_id = strftime('%Y%m%d-%H%M', gmtime())
    
    config = make_base_config()
    
    # Dataset configuration - SAME as before
    config["data"]["dataset_name"] = "gaussians"
    config["data"]["n_dims"] = 10
    config["data"]["data_args"] = {
        "n_samples": 10000,
        "n_dims": 10,
        "true_mutual_info": 5.0
    }
    config["data"]["data_dist_name"] = "gaussian"
    config["data"]["noise_dist_name"] = "gaussian"
    
    # ==================== IMPROVEMENTS ====================
    
    # 1. Better Architecture - Larger, deeper network
    config["architecture"]["network_type"] = "quadratic"
    config["architecture"]["quadratic_constraint_type"] = "symmetric_pos_diag"
    config["architecture"]["quadratic_head_use_linear_term"] = True
    config["architecture"]["mlp_width"] = 256  # INCREASED from 128
    config["architecture"]["n_mlp_layers"] = 4  # INCREASED from 3
    config["architecture"]["activation"] = "relu"
    
    # 2. Better Training Settings
    config["optimisation"]["n_epochs"] = 500  # INCREASED from 200
    config["optimisation"]["patience"] = 100  # INCREASED from 30
    config["optimisation"]["energy_lr"] = 5e-4  # DECREASED for stability
    config["optimisation"]["n_batch"] = 128
    config["optimisation"]["loss_function"] = "logistic"
    
    # 3. Better waymark spacing - More waymarks!
    configs_to_generate = []
    
    # Config 0: 8 waymarks (7 bridges) - VERY FINE approximation
    config0 = deepcopy(config)
    config0["data"]["linear_combo_alphas"] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    config0["data"]["initial_waymark_indices"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    configs_to_generate.append(("improved_8_waymarks", config0))
    
    # Config 1: 6 waymarks (5 bridges) - Good balance
    config1 = deepcopy(config)
    config1["data"]["linear_combo_alphas"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    config1["data"]["initial_waymark_indices"] = [0, 1, 2, 3, 4, 5]
    configs_to_generate.append(("improved_6_waymarks", config1))
    
    # Config 2: 4 waymarks (3 bridges) - Faster training
    config2 = deepcopy(config)
    config2["data"]["linear_combo_alphas"] = [0.0, 0.33, 0.67, 1.0]
    config2["data"]["initial_waymark_indices"] = [0, 1, 2, 3]
    configs_to_generate.append(("improved_4_waymarks", config2))
    
    # Save configurations
    for idx, (name, cfg) in enumerate(configs_to_generate):
        save_config(cfg, "improved", idx)
        print(f"Saved improved config {idx}: {name}")
        print(f"  Waymarks: {cfg['data']['initial_waymark_indices']}")
        print(f"  Network: {cfg['architecture']['n_mlp_layers']} layers × {cfg['architecture']['mlp_width']} units")
        print(f"  Training: {cfg['optimisation']['n_epochs']} epochs, patience={cfg['optimisation']['patience']}")
        print()
    
    print(f"Generated {len(configs_to_generate)} improved TDRE configurations")
    print("\nTo train improved TDRE models, run:")
    print("  python build_bridges.py --config_path=gaussians/improved/0  # Best (8 waymarks)")
    print("  python build_bridges.py --config_path=gaussians/improved/1  # Good (6 waymarks)")
    print("  python build_bridges.py --config_path=gaussians/improved/2  # Fast (4 waymarks)")
    print("\nRecommendation: Start with config 1 (6 waymarks) for best balance of speed/accuracy")


if __name__ == "__main__":
    print("Generating improved TDRE configurations...\n")
    make_improved_tdre_configs()
    print("\n" + "="*60)

