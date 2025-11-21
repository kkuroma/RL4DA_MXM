# Makefile for RL4DA_MXM project

.PHONY: remove_cache data_clean data_generate train_clean test train train_debug help

# Remove all __pycache__ directories recursively from base directory
remove_cache:
	@echo "Removing all __pycache__ directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache cleanup complete."

# Remove all data generation artifacts from logs directory (keeping config files)
data_clean:
	@echo "Cleaning generated data from logs directory..."
	find logs -name "trajectory.npy" -delete 2>/dev/null || true
	find logs -name "norm_dict.json" -delete 2>/dev/null || true
	find logs -type d -name "precomputed_paths" -exec rm -rf {} + 2>/dev/null || true
	find logs -type d -name "visualizations" -exec rm -rf {} + 2>/dev/null || true
	@echo "Data cleanup complete."

# Generate data for all L96 configurations in parallel
data_generate:
	@echo "Starting data generation for all L96 configurations in parallel..."
	@python scripts/generate_data_parallel.py --dir=logs
	@echo "Data generation complete for all configurations."

# Clean all training artifacts (TorchRL checkpoints, logs)
train_clean:
	@echo "Cleaning training artifacts from logs directory..."
	find logs -type d -name "checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find logs -type d -name "tensorboard" -exec rm -rf {} + 2>/dev/null || true
	find logs -name "best_weights.pt" -delete 2>/dev/null || true
	@echo "Training cleanup complete."

# Test the TorchRL environment (e.g., make test DIR=logs/L96_1)
test:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR parameter required. Usage: make test DIR=logs/L96_1"; \
		exit 1; \
	fi
	@echo "Running TorchRL environment tests for $(DIR)..."
	/home/kuroma/PyEnv/enkf_rl/bin/python test_torchrl_env.py --dir=$(DIR)

# Train multi-agent RL with TorchRL on a specific configuration (e.g., make train DIR=logs/L96_1)
train:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR parameter required. Usage: make train DIR=logs/L96_1"; \
		exit 1; \
	fi
	@echo "Starting TorchRL multi-agent training for $(DIR)..."; \
	/home/kuroma/PyEnv/enkf_rl/bin/python scripts/train.py --dir=$(DIR)

# Debug training with minimal steps (e.g., make train_debug DIR=logs/L96_1 STEPS=5)
train_debug:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR parameter required. Usage: make train_debug DIR=logs/L96_1 STEPS=5"; \
		exit 1; \
	fi
	@STEPS=$${STEPS:-5}; \
	echo "Running debug training for $(DIR) with $$STEPS steps..."; \
	/home/kuroma/PyEnv/enkf_rl/bin/python debug_train.py --dir=$(DIR) --steps=$$STEPS

# Show help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Data management:"
	@echo "  remove_cache         - Remove all __pycache__ directories recursively"
	@echo "  data_clean           - Remove generated data from logs (trajectory.npy, norm_dict.json, precomputed_paths/, visualizations/)"
	@echo "  data_generate        - Generate data for all L96 configurations in parallel"
	@echo ""
	@echo "Testing:"
	@echo "  test DIR=...         - Run TorchRL environment tests (e.g., make test DIR=logs/L96_1)"
	@echo ""
	@echo "Training:"
	@echo "  train DIR=...        - Train multi-agent RL with TorchRL (e.g., make train DIR=logs/L96_1)"
	@echo "  train_debug DIR=...  - Debug training with minimal steps (e.g., make train_debug DIR=logs/L96_1 STEPS=5)"
	@echo "  train_clean          - Remove training artifacts (checkpoints/, tensorboard/, best_weights.pt)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make data_generate                    # Generate data for all configs"
	@echo "  make test DIR=logs/L96_1              # Test environment"
	@echo "  make train_debug DIR=logs/L96_1       # Quick debug (5 steps)"
	@echo "  make train DIR=logs/L96_1 GPUS=1      # Full training with GPU"
	@echo "  make train_clean                      # Clean training artifacts"
	@echo ""
	@echo "  help                 - Show this help message"