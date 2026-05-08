# EM4CI_Python

Python implementation of **EM4CI** — an EM-based learning and causal inference pipeline for discrete Bayesian networks defined via XDSL model files.

## Overview

EM4CI learns CPT parameters for Bayesian networks with latent variables using Expectation-Maximisation (EM), then evaluates causal inference accuracy by comparing learned models against ground-truth models via interventional distributions $P(Y \mid do(X))$.

## Requirements

- Python ≥ 3.9
- See [requirements.txt](requirements.txt) for dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Learn a single model

```bash
python em4ci_core.py learn models_xdsl/em_15_cone_cloud_TD4_10_ED2_0.xdsl data/100/15_cone_cloud_TD4_10.csv 100
```

### Infer causal error (true vs learned model)

```bash
python em4ci_core.py infer models_xdsl/<true>.xdsl <learned>.xdsl Y --do X --num-samples 100
```

### Run full pipeline

```bash
# Predefined experiment set across sample sizes and domain sizes
python em4ci_core.py run --experiment-set small --sample-sizes 100 1000 --domains 2 4 --restarts 10

# Specific experiments
python em4ci_core.py run --experiments 15_cone_cloud_TD4_10 9_chain_TD4_10 --sample-sizes 100 --domains 2
```

### Key options for `run`

| Option | Default | Description |
|---|---|---|
| `--experiment-set` | `small` | Predefined set: `small` (8), `large` (6), `all` (14) |
| `--sample-sizes` | `100 1000` | Number of data samples |
| `--domains` | `2 4` | CPT domain sizes to evaluate |
| `--restarts` | `10` | EM random restarts per experiment |
| `--trials` | `10` | Independent trials (results averaged with ± std) |
| `--max-iter` | `100` | Max EM iterations per restart |
| `--force-learn` | off | Re-run learning even if cached results exist |
| `--name` | auto | Custom name for the output folder |

## Project Structure

```
em4ci_core.py        # Main pipeline script
models_xdsl/         # Ground-truth and template XDSL model files
data/
  100/               # CSV datasets with 100 samples
  1000/              # CSV datasets with 1000 samples
  10000/             # CSV datasets with 10000 samples
requirements.txt
```

## Output

Results are saved under `output/<run-name>/` as JSON summary files containing per-experiment metrics:
- `average_error` — mean absolute CPT error
- `weighted_error` — sample-weighted CPT error
- `ate_error` — average treatment effect error
