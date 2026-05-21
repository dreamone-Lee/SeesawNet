# SeesawNet

Official source code for the IJCAI 2026 accepted paper:

**SeesawNet: Towards Non-stationary Time Series Forecasting with Balanced Modeling of Common and Specific Dependencies**

## Overview

SeesawNet is a PyTorch implementation for long-term time series forecasting under non-stationary settings. The repository contains the main SeesawNet model, ablation variants, experiment code, and shell scripts for the benchmark datasets used in long-term forecasting.

## Requirements

The code was prepared for Python-based PyTorch training. Core dependencies include:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- matplotlib
- tqdm
- einops

Install dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if you plan to train on GPU.

## Dataset Preparation

Download the benchmark datasets and place them under `dataset/`.

The datasets can be obtained from the dataset links provided by the iTransformer repository:

- iTransformer repository: https://github.com/thuml/iTransformer
- Google Drive dataset archive: https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link

Expected examples:

```text
dataset/
  ETT-small/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  electricity/
    electricity.csv
  exchange_rate/
    exchange_rate.csv
  illness/
    national_illness.csv
  Solar/
    solar_AL.txt
  weather/
    weather.csv
```

## Usage

Clone the repository and enter the project directory:

```bash
git clone https://github.com/dreamone-Lee/SeesawNet.git
cd SeesawNet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full SeesawNet benchmark script:

```bash
sh scripts/SeesawNet.sh
```

You can also run a single dataset script from `scripts/run/`, for example:

```bash
sh scripts/run/etth1.sh SeesawNet SeesawNet_TFMAE 0 3 TFMAE
```

Arguments passed to dataset scripts are:

```text
model_name model_id gpu_id itr loss
```

Alternatively, run `run.py` directly with custom hyperparameters:

```bash
python -u run.py \
  --is_training 1 \
  --model_id SeesawNet_ETTh1 \
  --model SeesawNet \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --patch_len 32 \
  --stride 16 \
  --pd_layers 1 \
  --cr_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --dropout 0.4 \
  --down_sample_rate 1.0 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --loss TFMAE \
  --itr 3
```

## Repository Structure

```text
data_provider/   Dataset loading and preprocessing
experiments/     Training, validation, and testing loops
layers/          SeesawNet backbone layers and attention modules
model/           SeesawNet and ablation model definitions
scripts/         Reproduction scripts
utils/           Metrics, losses, time features, and training utilities
run.py           Main experiment entry point
```

## Outputs

Training creates runtime artifacts such as checkpoints, logs, CSV summaries, and test visualizations. These files are ignored by Git by default:

```text
checkpoints/
logs/
csv_results/
test_results/
results/
```

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{seesawnet2026,
  title     = {SeesawNet: Towards Non-stationary Time Series Forecasting with Balanced Modeling of Common and Specific Dependencies},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence},
  year      = {2026}
}
```

