10D Gaussian TDRE Experiment
============================

This note walks through the exact commands needed to reproduce the KL-analysis
experiment for 10D Gaussian pairs described in the recent updates. The workflow
matches the training/evaluation scripts already in the repo, so you can run
everything from scratch without touching the internals.

0. Setup
--------

```bash
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
conda env create -f ../environment.yml -n tre    # if you have not yet
conda activate tre
```

1. Generate Configs
-------------------

Create TDRE configs for all KL targets (10/15/20 nats), sample sizes
(50/100/300) and trials (default 10). Each config becomes a JSON in
`configs/gaussians_10d/model/` and encodes everything needed for training. The
generator now supports scaling the number of training epochs with sample size,
so we pass in breakpoints explicitly when we want longer training for larger
`n`.

```bash
python make_gaussians_10d_configs.py \
  --time_id=20251124-0137 \
  --waymark_breakpoints 1000 \
  --waymark_counts 12 12 \
  --epoch_breakpoints 1000 \
 --epoch_values 400 500
```

Adjust `--time_id` to any string you prefer. This identifier gets embedded in
the save paths under `saved_models/tre_gaussians_10d/<time_id>_<idx>/`
(override via `--save_root` if you want the legacy name).
For the KL-sweep experiment (plotting error vs. target KL), simply pass
`--target_kls 5 10 20 30 40` (or any list you like) when running the command
above so the generator emits configs for each desired KL gap.

2. Train TDRE models
--------------------

Loop over every JSON and train a model via `build_bridges.py`. The snippet below
runs sequentially; feel free to parallelize if you have a cluster or multiple
GPUs.

```bash
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
for cfg in configs/gaussians_10d/model/*.json; do
    idx=$(basename "${cfg%.json}")
    echo "Training config ${idx}"
    python build_bridges.py --config_path="gaussians_10d/model/${idx}"
done
```

Each run writes checkpoints/metrics under
`saved_models/tre_gaussians_10d/20251124-0137_<idx>/`.

3. Train BDRE models
--------------------

Train matching BDRE classifiers for every config so the two methods can be
compared on identical datasets/time-ids. We now construct BDRE datasets using the
same Gaussian sampler (and `data_seed`) as TDRE, so both methods see the exact
same numerator samples for each config. This writes checkpoints to
`saved_models/bdre_gaussians_10d/<time_id>_<idx>/`.

```bash
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
for cfg in configs/gaussians_10d/model/*.json; do
    idx=$(basename "${cfg%.json}")
    echo "Training BDRE config ${idx}"
    python train_bdre_gaussians_10d.py --config_path="gaussians_10d/model/${idx}"
done
```

4. Train DV (Donsker–Varadhan) models
-------------------------------------

Train a PyTorch DV estimator on the same data splits.

```bash
pip install --no-deps torch==2.1.0  # install once inside the 'dre' environment
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
for cfg in configs/gaussians_10d/model/*.json; do
    idx=$(basename "${cfg%.json}")
    echo "Training DV config ${idx}"
    python train_dv_gaussians_10d.py --config_path="gaussians_10d/model/${idx}"
done
```

All three estimators share the exact same numerator samples (and consistent
denominator draws) thanks to the common `data_seed` and GAUSSIANS sampler.

This saves to `saved_models/dv_gaussians_10d/<time_id>_<idx>/`.

5. Train NWJ (Nguyen–Wainwright–Jordan) models
----------------------------------------------

Train an NWJ KL lower-bound estimator on the same data splits.

```bash
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
for cfg in configs/gaussians_10d/model/*.json; do
    idx=$(basename "${cfg%.json}")
    echo "Training NWJ config ${idx}"
    python train_nwj_gaussians_10d.py --config_path="gaussians_10d/model/${idx}"
done
```

6. Train MDRE (Multinomial Density Ratio Estimation) models
----------------------------------------------------------

The MDRE classifier jointly discriminates numerator, denominator, and all
waymark mixtures with a multinomial logistic head. We recover the log density
ratio by subtracting the logits for the first two classes. Training uses the
exact same cached samples as the other estimators and produces checkpoints under
`saved_models/mdre_gaussians_10d/<time_id>_<idx>/`.

```bash
cd /Users/johnstarlu/Desktop/CMU/Research/TRE__Code/tre_code
for cfg in configs/gaussians_10d/model/*.json; do
    idx=$(basename "${cfg%.json}")
    echo "Training MDRE config ${idx}"
    python train_mdre_gaussians_10d.py --config_path="gaussians_10d/model/${idx}"
done
```

7. Evaluate & Plot
------------------

After all trainings finish, aggregate the KL estimates and draw the curve of
relative errors versus training sample size. This evaluates every saved run
(TDRE only, BDRE only, or both), averages across the trials per sample size, and
contrasts multiple evaluation batch sizes if desired.

```bash
python evaluate_gaussians_10d.py \
    --tdre_model_base tre_gaussians_10d/20251124-0137 \
    --bdre_model_base bdre_gaussians_10d/20251124-0137 \
    --dv_model_base dv_gaussians_10d/20251124-0137 \
    --nwj_model_base nwj_gaussians_10d/20251124-0137 \
    --mdre_model_base mdre_gaussians_10d/20251124-0137 \
    --eval_sizes 10 100 1000 \
    --n_eval_trials 10 \
    --sample_sizes 50 100 300 500 1000 1500 2000 \
    --kl_targets 20 \
    --save_plot results/gauss10d/tdre_bdre_dv_nwj.png \
    --save_plot_pdf results/gauss10d/tdre_bdre_dv_nwj.pdf \
    --save_plot_zoom results/gauss10d/tdre_bdre_dv_nwj_zoom.png \
    --save_plot_zoom_pdf results/gauss10d/tdre_bdre_dv_nwj_zoom.pdf \
    --save_summary results/gauss10d/eval_summary.txt
```

- `--tdre_model_base` / `--bdre_model_base` / `--dv_model_base` / `--nwj_model_base` / `--mdre_model_base` select directories matching
  `saved_models/<dataset>/<time_id>_*`. Drop any you didn’t train.
- `--eval_sizes` lets you compare evaluation sample sizes (e.g. 10 vs 100 vs 1000).
- The script prints per-model relative errors, averages them per
  `(true_KL, training_n)`, writes both a full-range plot and a zoomed plot
  (n ≥ 1000), and dumps a text summary (`eval_summary.txt`) with all settings.

Results appear on screen and can be exported via Matplotlib if needed.

8. KL Sweep Plot (relative error vs. true KL)
---------------------------------------------

After training models for multiple KL targets (e.g. 5/10/20/30/40) you can
generate the additional plot requested—x-axis = analytic KL, y-axis = relative
error at a fixed training sample size and evaluation batch size (M = 1000 by
default):

```bash
python evaluate_gaussians_kl_sweep.py \
    --tdre_model_base tre_gaussians_10d/20251124-0137 \
    --bdre_model_base bdre_gaussians_10d/20251124-0137 \
    --dv_model_base dv_gaussians_10d/20251124-0137 \
    --nwj_model_base nwj_gaussians_10d/20251124-0137 \
    --mdre_model_base mdre_gaussians_10d/20251124-0137 \
    --target_kls 5 10 20 30 40 \
    --sample_size_focus 1000 \
    --eval_size 1000 \
    --save_plot results/gauss10d/kl_sweep.png \
    --save_plot_pdf results/gauss10d/kl_sweep.pdf \
    --save_summary results/gauss10d/kl_sweep_summary.txt
```

Adjust `--sample_size_focus` if you want to analyze a different training sample
count; it must match one of the `--sample_sizes` provided during config
generation. The script shares the same cached validation data with the main
evaluation file so all estimators are compared on identical evaluation batches.

Notes
-----
- You can lower `--n_trials` or prune the `--target_kls/--sample_sizes` when
  generating configs if you want a smaller sweep.
- For clusters, wrap the `build_bridges.py` command in a Slurm array job or use
  `run_pipeline.py`. The JSON configs do not depend on execution order.

That’s everything—generate configs once, loop over them to train, and run the
aggregated evaluation/plotting command.***
