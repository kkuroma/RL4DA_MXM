# Makefile for RL4DA_MXM Training Pipeline

# Default target
.DEFAULT_GOAL := help

# Phony targets (not files)
.PHONY: help clear reset train tensorboard restart_training end_training status visualize

# Help target - shows available commands
help:
	@echo "Available targets:"
	@echo "  clear           - Delete all training results directories"
	@echo "  reset           - Nuke all logs except config files (complete reset)"
	@echo "  train           - Run concurrent training for all L96 configurations"
	@echo "  tensorboard     - Launch TensorBoard to view training logs"
	@echo "  restart_training - Clear results, then run train and tensorboard in background"
	@echo "  end_training    - Stop background train and tensorboard processes"
	@echo "  status          - Check if training processes are running"
	@echo "  visualize       - Create visualizations for all precomputed data"
	@echo ""
	@echo "Usage examples:"
	@echo "  make clear          # Clean up old training results"
	@echo "  make reset          # Complete reset - remove all logs except configs"
	@echo "  make restart_training # Start fresh training session"
	@echo "  make status         # Check training status"
	@echo "  make end_training   # Stop training processes"
	@echo "  make visualize      # Create visualizations for precomputed data"

# Clear target - delete training results
clear:
	@echo "Cleaning up training results..."
	@rm -rf logs/L96_1/training_results
	@rm -rf logs/L96_2/training_results
	@rm -rf logs/L96_3/training_results
	@echo "Training results directories cleared."

# Reset target - nuke all logs except config files
reset:
	@echo "🚨 RESETTING: Nuking all logs except config files..."
	@echo "Preserving config files and removing all other data..."
	@find logs/L96_1 -type f ! -name "config.py" -delete 2>/dev/null || true
	@find logs/L96_2 -type f ! -name "config.py" -delete 2>/dev/null || true
	@find logs/L96_3 -type f ! -name "config.py" -delete 2>/dev/null || true
	@find logs/L96_1 -type d -empty -delete 2>/dev/null || true
	@find logs/L96_2 -type d -empty -delete 2>/dev/null || true
	@find logs/L96_3 -type d -empty -delete 2>/dev/null || true
	@rm -f logs/L96_1/__pycache__/* 2>/dev/null || true
	@rm -f logs/L96_2/__pycache__/* 2>/dev/null || true
	@rm -f logs/L96_3/__pycache__/* 2>/dev/null || true
	@rmdir logs/L96_1/__pycache__ 2>/dev/null || true
	@rmdir logs/L96_2/__pycache__ 2>/dev/null || true
	@rmdir logs/L96_3/__pycache__ 2>/dev/null || true
	@echo "✅ Reset complete! Only config files remain in logs."

# Train target - run concurrent training
train:
	@echo "Starting concurrent training for all L96 configurations..."
	@./scripts/train_concurrent.sh

# TensorBoard target - launch TensorBoard server
tensorboard:
	@echo "Launching TensorBoard..."
	@./launch_tensorboard.sh

# Restart training target - clear results and start training/tensorboard in background
restart_training:
	@echo "Restarting training pipeline..."
	@$(MAKE) clear
	@echo "Starting training and tensorboard in background..."
	@nohup $(MAKE) train > /tmp/rl4da_train.log 2>&1 & echo $$! > /tmp/rl4da_train.pid
	@nohup $(MAKE) tensorboard > /tmp/rl4da_tensorboard.log 2>&1 & echo $$! > /tmp/rl4da_tensorboard.pid
	@echo "Training started with PID: $$(cat /tmp/rl4da_train.pid)"
	@echo "TensorBoard started with PID: $$(cat /tmp/rl4da_tensorboard.pid)"
	@echo "Use 'make status' to check process status or 'make end_training' to stop"

# End training target - stop background training processes
end_training:
	@echo "Stopping training processes..."
	@if [ -f /tmp/rl4da_train.pid ]; then \
		if kill -0 $$(cat /tmp/rl4da_train.pid) 2>/dev/null; then \
			kill $$(cat /tmp/rl4da_train.pid) && echo "Training process stopped"; \
		else \
			echo "Training process not running"; \
		fi; \
		rm -f /tmp/rl4da_train.pid; \
	else \
		echo "No training PID file found"; \
	fi
	@if [ -f /tmp/rl4da_tensorboard.pid ]; then \
		if kill -0 $$(cat /tmp/rl4da_tensorboard.pid) 2>/dev/null; then \
			kill $$(cat /tmp/rl4da_tensorboard.pid) && echo "TensorBoard process stopped"; \
		else \
			echo "TensorBoard process not running"; \
		fi; \
		rm -f /tmp/rl4da_tensorboard.pid; \
	else \
		echo "No TensorBoard PID file found"; \
	fi

# Status target - check if training processes are running
status:
	@echo "Training process status:"
	@if [ -f /tmp/rl4da_train.pid ]; then \
		if kill -0 $$(cat /tmp/rl4da_train.pid) 2>/dev/null; then \
			echo "  Training: RUNNING (PID: $$(cat /tmp/rl4da_train.pid))"; \
		else \
			echo "  Training: STOPPED (stale PID file)"; \
			rm -f /tmp/rl4da_train.pid; \
		fi; \
	else \
		echo "  Training: NOT RUNNING"; \
	fi
	@if [ -f /tmp/rl4da_tensorboard.pid ]; then \
		if kill -0 $$(cat /tmp/rl4da_tensorboard.pid) 2>/dev/null; then \
			echo "  TensorBoard: RUNNING (PID: $$(cat /tmp/rl4da_tensorboard.pid))"; \
		else \
			echo "  TensorBoard: STOPPED (stale PID file)"; \
			rm -f /tmp/rl4da_tensorboard.pid; \
		fi; \
	else \
		echo "  TensorBoard: NOT RUNNING"; \
	fi

# Visualize target - create visualizations for all precomputed data
visualize:
	@echo "Creating visualizations for all configurations..."
	@for config_dir in logs/L96_*; do \
		if [ -d "$$config_dir" ]; then \
			echo "Creating visualizations for $$config_dir..."; \
			/home/kuroma/PyEnv/enkf_rl/bin/python scripts/create_visualizations.py --config_dir "$$config_dir"; \
		fi; \
	done
	@echo "✅ Visualizations created for all configurations!"