#!/bin/bash

# TensorBoard launcher script
# Launches TensorBoard to view logs from all three L96 training experiments

# Set the port for TensorBoard
PORT=6006
HOST="0.0.0.0"

# Define the log directories
LOG_DIR1="logs/L96_1/training_results/tensorboard"
LOG_DIR2="logs/L96_2/training_results/tensorboard"
LOG_DIR3="logs/L96_3/training_results/tensorboard"

echo "=== TensorBoard Launcher ==="
echo "Starting TensorBoard server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo ""
echo "Log directories:"
echo "  L96_1: $LOG_DIR1"
echo "  L96_2: $LOG_DIR2"
echo "  L96_3: $LOG_DIR3"
echo ""

# Check if directories exist
missing_dirs=()
for dir in "$LOG_DIR1" "$LOG_DIR2" "$LOG_DIR3"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "Warning: The following log directories don't exist yet:"
    for dir in "${missing_dirs[@]}"; do
        echo "  - $dir"
    done
    echo ""
    echo "TensorBoard will still start and will pick up logs as they become available."
    echo ""
fi

# Launch TensorBoard with multiple log directories
echo "Launching TensorBoard..."
echo "Access TensorBoard at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

/home/kuroma/PyEnv/enkf_rl/bin/tensorboard --logdir_spec=L96_1:$LOG_DIR1,L96_2:$LOG_DIR2,L96_3:$LOG_DIR3 \
            --host=$HOST \
            --port=$PORT \
            --reload_multifile=true \
            --reload_interval=30