# One Model, Two Roles: Emergent Specialization in a Shared Recurrent Transformer

This repository provides the **official PyTorch implementation** of our paper:

> **"One Model, Two Roles: Emergent Specialization in a Shared Recurrent Transformer"**<br>
> *[Jucheng Shen](https://juchengshen.github.io/)\*, [Wenyi Su](https://www.linkedin.com/in/barbara-su-966314268/)\*, [Anastasios Kyrillidis](https://akyrillidis.github.io/about/)*<br>
> [arXiv:2605.17811](https://arxiv.org/abs/2605.17811)

AIR studies how a shared recurrent transformer can develop specialized low-level and high-level computation roles while keeping a single set of model parameters. This codebase contains the AIR model variants, Sudoku-Extreme and Maze training scripts, and the analysis utilities used for the paper experiments.

<div align="center">
  <img src="assets/air_architecture_diagram.png" alt="AIR architecture overview" width="700"/>
</div>

## News

- May 2026: Initial AIR code release with training, ablation, visualization, and attention-analysis scripts.

## Environment Setup

Python 3.10+ is recommended. Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

### AdamATan2 Optimizer

AdamATan2 is bundled as `adam_atan2.zip` instead of a Git submodule or PyPI dependency. Unzip it once from the repository root before training:

```bash
unzip adam_atan2.zip
```

This creates `adam_atan2/`, which exposes the `AdamATan2` optimizer used by `pretrain.py`. If you want to refresh the local copy, remove the extracted directory and unzip again:

```bash
rm -rf adam_atan2
unzip adam_atan2.zip
```

You do not need to initialize an `adam_atan2` submodule or install `adam-atan2` with pip.

This project uses PyTorch with CUDA and builds CUDA extensions. If CUDA 12.6 and the matching PyTorch wheels are not already installed, one tested setup is:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override
export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Packages used when building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

The training scripts log to [Weights & Biases](https://wandb.ai/) and call `wandb login "${WANDB_API_KEY}"`, so set your API key before launching experiments:

```bash
export WANDB_API_KEY=your_wandb_api_key
```

## Datasets

Build the Sudoku-Extreme and Maze datasets:

```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

python dataset/build_maze_dataset.py \
  --output-dir data/maze-30x30-hard-1k
```

If you already have prebuilt datasets, place them at:

- `data/sudoku-extreme-1k-aug-1000`
- `data/maze-30x30-hard-1k`

## Usage

All experiment scripts use repository-relative data, checkpoint, and output paths by default, so they can be run directly from the repository root after the environment is activated.

### Training And Ablations

The `run_all_*.sh` scripts submit the full sweeps with `sbatch`:

```bash
bash experiment_input-injection-specialization-sudoku/run_all_input_injection_specialization.sh
bash experiment_input-injection-specialization-maze/run_all_input_injection_specialization.sh
bash experiment_operator-form-control/run_all_operator_form_control.sh
bash experiment_addition-prepend-strip-no-strip/run_all_addition_prepend_strip_no_strip.sh
```

If you are not using Slurm, run the individual training scripts directly instead:

```bash
bash experiment_input-injection-specialization-sudoku/train_sudoku_Lx_H.sh
bash experiment_input-injection-specialization-maze/train_maze_Lx_H.sh
```

The input-injection scripts use the paper naming convention for AIR variants, including `L_Hx`, `Lx_H`, `L_H2x`, `L2x_H`, `Lx_H2x`, `L2x_Hx`, `Lx_Hx`, and `L2x_H2x`.

### Visual Freeze And Decode Experiments

Run the visual experiment scripts in:

- `experiment_visual-sudoku-decoded-freeze/`
- `experiment_visual-maze-decoded-freeze/`

The `*_freeze_*.sh` and `decode_*_intermediate_first_10.sh` scripts require trained checkpoints before they can run. By default they look for checkpoint files and the matching `all_config.yaml` under the repo-local `checkpoints/` paths written in each script. Edit the checkpoint path in the script, or set the corresponding environment variable such as `AIR_SUDOKU_CKPT_PATH` or `AIR_MAZE_CKPT_PATH`, if your checkpoint is stored elsewhere.

### Attention Analysis

Regenerate the bar-chart data and figures with:

```bash
bash experiment_attention-analysis-sudoku/generate_bar_data.sh
bash experiment_attention-analysis-sudoku/multilayer_figure.sh

bash experiment_attention-analysis-maze/generate_bar_data.sh
bash experiment_attention-analysis-maze/multilayer_figure.sh
```

`generate_bar_data.py` captures L/H attention maps over 1,000 test puzzles at sub-steps `{2,4,6,8,10,12,14,15}` and writes per-layer JSON files into `bar_data/`.

## Repository Layout

- `models/air/`: AIR architecture variants.
- `adam_atan2.zip`: bundled AdamATan2 optimizer package; unzip locally before training.
- `config/`: default pretraining and architecture configs.
- `dataset/`: Sudoku-Extreme and Maze dataset builders.
- `experiment_input-injection-specialization-*`: AIR asymmetry and symmetry ablations.
- `experiment_operator-form-control/`: operator-form control experiments.
- `experiment_addition-prepend-strip-no-strip/`: input-token ablations.
- `experiment_visual-*-decoded-freeze/`: freeze and intermediate-decoding experiments.
- `experiment_attention-analysis-*/`: attention statistic and heatmap generation.

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{shen2026modelrolesemergentspecialization,
      title={One Model, Two Roles: Emergent Specialization in a Shared Recurrent Transformer},
      author={Jucheng Shen and Barbara Su and Anastasios Kyrillidis},
      year={2026},
      eprint={2605.17811},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.17811},
}
```
