# RL4DA_MXM: Multi-Agent RL for Data Assimilation

Reinforcement Learning for Data Assimilation using Multi-Agent Ensemble Kalman Filter

## Overview

This project implements a **multi-agent reinforcement learning** approach to optimize Ensemble Kalman Filter (EnKF) data assimilation. Instead of a single agent controlling all N×N_ens dimensions, we deploy **N_ens independent agents**, where each agent controls its own ensemble member (N dimensions).

### Key Innovation

- **Problem**: Single-agent action space grows to N×N_ens (e.g., 20×20 = 400 dimensions), making learning intractable
- **Solution**: Each of N_ens agents predicts one ensemble member (N dimensions), reducing per-agent action space
- **Reward**: Individual agents receive rewards based on their ensemble member's RMSE: `reward = 1 - RMSE`
- **Total reward**: Averaged across all agents

## Architecture

### Multi-Agent Environment (`MultiAgentEnkfEnvironment`)

Each agent receives an observation containing:
- `xa_prev_i` (N dims): Previous analysis state for ensemble member i
- `xb_i` (N dims): Background state for ensemble member i
- `dxa_i` (N dims): Derivatives for ensemble member i
- `xo` (N dims): Observation (shared across agents)

**Total observation per agent**: 4×N dimensions

**Action per agent**: N dimensions (the predicted analysis state for that ensemble member)

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- Ray[RLlib] >= 2.9.0
- PyTorch >= 2.0.0
- Gymnasium >= 0.28.0
- NumPy, SciPy, Matplotlib

## Usage

### 1. Generate Training Data

First, generate precomputed EnKF trajectories:

```bash
make data_generate
```

This creates precomputed paths in `logs/L96_*/precomputed_paths/` for all configurations.

### 2. Train Multi-Agent RL

Train on a specific configuration:

```bash
make train DIR=logs/L96_1
```

Or use the Python script directly for more control:

```bash
python scripts/train_multiagent.py --dir=logs/L96_1 --num-workers=4 --num-gpus=0
```

**Options**:
- `--dir`: Log directory (required)
- `--num-workers`: Number of parallel rollout workers (default: 2)
- `--num-gpus`: Number of GPUs to use (default: 0.0)

### 3. Hyperparameter Tuning

Run automated hyperparameter search with Ray Tune:

```bash
make tune DIR=logs/L96_1 SAMPLES=20
```

Or:

```bash
python scripts/train_multiagent.py --dir=logs/L96_1 --tune --num-samples=20 --max-concurrent-trials=2
```

This will search over:
- Learning rates: log-uniform(1e-5, 1e-3)
- Discount factors: {0.95, 0.98, 0.99, 0.995}
- GAE lambda: {0.9, 0.95, 0.98}
- Batch sizes, minibatch sizes, SGD iterations, clip parameters, etc.

Results are saved to `logs/L96_1/ray_results/`.

### 4. Test Environment

Verify the environment works correctly:

```bash
python test_multiagent_env.py
```

## Project Structure

```
RL4DA_MXM/
├── src/
│   ├── rl/
│   │   ├── env.py              # MultiAgentEnkfEnvironment
│   │   └── __init__.py
│   ├── models/
│   │   ├── l96.py              # Lorenz 96 model
│   │   ├── l63.py              # Lorenz 63 model
│   │   └── __init__.py
│   └── enkf/
│       ├── solver.py           # EnKF solver
│       └── __init__.py
├── scripts/
│   ├── generate_data.py        # Single config data generation
│   ├── generate_data_parallel.py # Parallel data generation
│   └── train_multiagent.py     # Multi-agent training with Ray
├── logs/
│   ├── global_config.py        # Global hyperparameters
│   ├── L96_1/config.py         # Config for L96 with N=20
│   ├── L96_2/config.py         # Config for L96 with N=40
│   └── L96_3/config.py         # Config for L96 with N=60
├── Makefile                    # Convenient commands
├── requirements.txt            # Python dependencies
├── test_multiagent_env.py      # Environment test script
└── README.md                   # This file
```

## Configuration

### Global Configuration (`logs/global_config.py`)

Controls EnKF parameters and RL hyperparameters:

```python
config = {
    "num_ensembles": 20,        # Number of ensemble members (agents)
    "oda": 1.0,                 # Observation window
    "dtda": 2.5e-3,             # Integration timestep

    "rl_multiagent": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        # ... more parameters
    }
}
```

### Model-Specific Configuration (`logs/L96_*/config.py`)

Inherits from global config and specifies model details:

```python
config = {
    **global_config.config,
    "model_class": L96,
    "params": {"N": 20, "F": 8.0},
}
```

## Makefile Commands

```bash
make help              # Show all available commands
make remove_cache      # Remove __pycache__ directories
make data_clean        # Remove generated data
make data_generate     # Generate training data for all configs
make train_clean       # Remove training artifacts
make train DIR=...     # Train multi-agent RL
make tune DIR=... SAMPLES=N  # Hyperparameter tuning
```

## Outputs

### Training Outputs
- **Checkpoints**: `logs/L96_*/checkpoints_ma/checkpoint_XXXXX`
- **Best model**: `logs/L96_*/models_ma/final_model`
- **Logs**: `logs/L96_*/run_multiagent.log`

### Tuning Outputs
- **Ray results**: `logs/L96_*/ray_results/`
- **Best hyperparameters**: Logged in `run_multiagent.log`

## Monitoring Training

Ray provides a web dashboard at `http://localhost:8265` when training is running.

You can also monitor via logs:

```bash
tail -f logs/L96_1/run_multiagent.log
```

## Citations

If you use this code, please cite the relevant papers for:
- Ensemble Kalman Filters
- Multi-Agent Reinforcement Learning
- Ray/RLlib

## License

[Specify your license here]
