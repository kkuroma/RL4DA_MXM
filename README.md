# RL4DA_MXM: Reinforcement Learning for Data Assimilation

## Overview

This project implements reinforcement learning agents to predict Ensemble Kalman Filter (ENKF) analysis states in dynamical systems. Instead of using traditional ENKF algorithms, RL agents learn to produce optimal analysis corrections by observing temporal patterns in forecast errors and ensemble dynamics.

### Key Features

- **Temporal LSTM Architecture**: Respects physics causality with proper temporal sequencing
- **Multiple L96 Configurations**: Support for Lorenz-96 systems with different parameters
- **Robust Normalization**: Handles extreme values with proper (-3, 3) range scaling
- **Curriculum Learning**: Progressive training difficulty with path-based curriculum
- **Comprehensive Logging**: TensorBoard integration with detailed metrics
- **Data Quality Verification**: Ensures ENKF improvement and data consistency

## Project Structure

```
RL4DA_MXM/
├── src/                          # Source code
│   ├── envs/                     # RL environments
│   │   └── enkf_env.py          # Main ENKF RL environment
│   ├── models/                   # Physics models
│   │   ├── l96.py               # Lorenz-96 model
│   │   └── l63.py               # Lorenz-63 model
│   ├── enkf/                     # ENKF implementation
│   │   └── eakf_solver.py       # Ensemble Adjusted Kalman Filter
│   ├── agents/                   # RL agent architectures
│   │   ├── temporal_lstm.py     # Temporal LSTM features extractor
│   │   └── policies.py          # Custom SB3 policies
│   ├── utils/                    # Utilities
│   │   ├── normalization.py     # Data normalization
│   │   └── data_generation.py   # Trajectory and path generation
│   └── training/                 # Training infrastructure
│       ├── trainer.py           # Main training script
│       └── callbacks.py         # Training callbacks
├── logs/                         # Training data and results
│   ├── global_config.py         # Shared configuration
│   ├── L96_1/                   # N=20, F=5.0 (ground truth)
│   ├── L96_2/                   # N=17, F=5.0
│   └── L96_3/                   # N=20, F=4.5
├── scripts/                      # Shell scripts
│   └── train_concurrent.sh      # Concurrent training script
├── Makefile                      # Build and training commands
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Temporal LSTM Architecture

### Key Design Principles

1. **Respect Physics Causality**: No time-travel in data flow
2. **Temporal vs Simultaneous Processing**:
   - Sequential: Previous analysis states, forecast errors, ensemble spread
   - Simultaneous: Current ensemble forecast, observations, truth
3. **Proper LSTM Input**: Only genuine temporal dependencies

### Data Flow

```
Timestep t-2: [analysis, forecast_error, ensemble_spread] ┐
Timestep t-1: [analysis, forecast_error, ensemble_spread] ├─► LSTM ─┐
                                                          ┘         │
Current t:    [forecast_ensemble, truth, observations,            │
               ensemble_mean, ensemble_std] ──────────────────────┼─► Combine ─► Analysis Prediction
                                                                  │
                                                                  ┘
```

## Installation

1. **Create Python Environment**:
   ```bash
   python -m venv /home/kuroma/PyEnv/enkf_rl
   source /home/kuroma/PyEnv/enkf_rl/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python -c "import torch; import stable_baselines3; print('Installation successful')"
   ```

## Configuration

### L96 System Parameters

- **L96_1**: N=20, F=5.0 (canonical parameters, used as ground truth)
- **L96_2**: N=17, F=5.0 (different dimension)
- **L96_3**: N=20, F=4.5 (different forcing)

### Key Configuration Parameters

```python
# ENKF Parameters
"num_ensembles": 20,           # Ensemble size
"dtda": 2.5e-3,               # Time step size
"oda": 1.0,                   # Time between observations
"inflation": 5.0,              # Multiplicative inflation

# Temporal Architecture
"temporal_window": 3,          # Number of timesteps for LSTM
"lstm_hidden_size": 512,       # LSTM hidden state size
"num_lstm_layers": 2,          # LSTM depth

# Training Parameters
"learning_rate": 3e-4,         # Conservative for LSTM stability
"batch_size": 64,              # Smaller batches for LSTM
"ent_coef": 0.01,             # Small entropy for precision tasks
"max_grad_norm": 0.5,         # Critical for LSTM stability

# Curriculum Learning
"enable_curriculum": True,
"curriculum_start_paths": 1,   # Start with 1 training path
"curriculum_max_paths": 99,    # Maximum paths
"curriculum_episodes_per_path": 100,  # Episodes needed per level
```

## Usage

### Training a Single Configuration

```bash
# Quick test (10k timesteps)
python src/training/trainer.py --config_dir logs/L96_1 --test

# Full training
python src/training/trainer.py --config_dir logs/L96_1

# Force data regeneration
python src/training/trainer.py --config_dir logs/L96_1 --force-regenerate
```

### Concurrent Training (All Configurations)

```bash
# Using Makefile
make train

# Or directly
./scripts/train_concurrent.sh
```

### Makefile Commands

```bash
make help           # Show available commands
make clear          # Clear training results
make reset          # Complete reset (keep configs only)
make train          # Run concurrent training
make tensorboard    # Launch TensorBoard
```

### Monitoring Training

1. **TensorBoard**:
   ```bash
   tensorboard --logdir logs/L96_1/training_results/tensorboard
   ```

2. **Key Metrics to Watch**:
   - `episode/reward`: Should increase over time
   - `episode/rmse_mean_10`: Should decrease
   - `curriculum/current_level`: Curriculum progression
   - `gradients/lstm_total_norm`: Should be stable (~0.1-1.0)
   - `policy/action_std_mean`: Exploration level

## Data Generation

### Source Trajectory
- 1M timesteps with 100k warmup steps
- Samples from stabilized region (last 500k steps)
- Quality verification ensures convergence

### ENKF Paths
- 100 precomputed trajectories per configuration
- 1000 timesteps per path
- Verified ENKF improvement (analysis RMSE < forecast RMSE)

### Normalization Strategy
- Target range: [-3, 3] with buffer to [-3.1, 3.1]
- Percentile-based normalization (1st-99th percentile)
- Separate factors for each data type
- Proper unnormalization for ENKF integration

## Troubleshooting

### Common Issues

1. **Entropy Collapse (-500 entropy)**:
   - Check action_std values in TensorBoard
   - Ensure proper policy initialization (`log_std_init=0.0`)
   - Verify gradient norms are reasonable

2. **LSTM Gradient Issues**:
   - Monitor `gradients/lstm_total_norm`
   - Ensure `max_grad_norm=0.5` is set
   - Check for proper weight initialization

3. **Environment Instability**:
   - Verify normalization ranges
   - Check early termination thresholds
   - Ensure action clipping to [-3, 3]

4. **Poor ENKF Performance**:
   - Verify data quality with `verify_data_quality()`
   - Check inflation parameters
   - Ensure proper ensemble initialization

### Performance Optimization

1. **GPU Usage**:
   ```python
   config["use_cuda"] = True
   ```

2. **Memory Management**:
   - Reduce `batch_size` if OOM
   - Lower `temporal_window` if needed
   - Use `n_envs=1` for curriculum learning

3. **Training Speed**:
   - Increase `n_steps` for better GPU utilization
   - Use vectorized environments for simple tasks
   - Profile with `torch.profiler` if needed

## File Format

### Precomputed Paths
```python
path = {
    'true_states': np.ndarray,      # Shape: (T, N)
    'forecast_states': np.ndarray,  # Shape: (T, N, N_ens)
    'analysis_states': np.ndarray,  # Shape: (T, N, N_ens)
    'observations': np.ndarray,     # Shape: (T, N_obs)
    'start_index': int,             # Original trajectory index
    'path_id': int                  # Path identifier
}
```

### Normalization Factors
```json
{
    "true_states": {"min": float, "max": float, "range": float},
    "forecast_states": {"min": float, "max": float, "range": float},
    ...
}
```

## Research Notes

### Why Temporal LSTM?
- ENKF has inherent temporal dependencies
- Recent forecast errors inform current corrections
- Ensemble uncertainty evolution follows temporal patterns
- Standard MLPs cannot capture sequential error dynamics

### Curriculum Learning Strategy
- Start with 1 training path (simple)
- Add paths as success rate ≥ 70%
- Episodes needed per level: 100/n (where n = current level)
- Prevents overfitting to specific initial conditions

### Reward Design
- Primary: RMSE-based reward in normalized space
- Range: [-1, 1] with good predictions getting positive rewards
- Bonus: Small improvement reward for error reduction
- Early termination: Prevents unstable trajectories

## Contributing

1. Follow existing code structure and naming conventions
2. Add comprehensive docstrings for new functions
3. Include unit tests for new functionality
4. Update this README for significant changes
5. Verify data quality for new physics models

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rl4da_mxm,
    title={Reinforcement Learning for Data Assimilation in Dynamical Systems},
    author={Your Name},
    year={2025},
    url={https://github.com/your-repo/RL4DA_MXM}
}
```