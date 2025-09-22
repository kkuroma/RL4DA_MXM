#!/bin/bash

# Concurrent training script for L96 configs
# Runs trainer.py on three L96 configs in parallel

# Function to run training with proper error handling
run_training() {
    local config_name=$1
    local config_dir="logs/${config_name}"
    
    echo "Starting training for ${config_name}..."
    /home/kuroma/PyEnv/enkf_rl/bin/python src/training/trainer.py --config_dir "$config_dir" 2>&1 | tee "logs/${config_name}_training.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "${config_name} training completed successfully"
    else
        echo "${config_name} training failed with exit code ${PIPESTATUS[0]}"
    fi
}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting concurrent training for L96 configs..."
echo "Logs will be saved to logs/ directory"

# Start training processes in background
run_training "L96_1" &
PID1=$!

run_training "L96_2" &
PID2=$!

run_training "L96_3" &
PID3=$!

# Store PIDs for monitoring
echo "Training processes started:"
echo "L96_1: PID $PID1"
echo "L96_2: PID $PID2"
echo "L96_3: PID $PID3"

# Wait for all processes to complete
echo "Waiting for all training processes to complete..."

wait $PID1
EXIT1=$?

wait $PID2
EXIT2=$?

wait $PID3
EXIT3=$?

# Report results
echo "=== Training Results ==="
echo "L96_1: $([ $EXIT1 -eq 0 ] && echo "SUCCESS" || echo "FAILED (exit code: $EXIT1)")"
echo "L96_2: $([ $EXIT2 -eq 0 ] && echo "SUCCESS" || echo "FAILED (exit code: $EXIT2)")"
echo "L96_3: $([ $EXIT3 -eq 0 ] && echo "SUCCESS" || echo "FAILED (exit code: $EXIT3)")"

# Exit with error if any training failed
if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ] || [ $EXIT3 -ne 0 ]; then
    echo "One or more training processes failed!"
    exit 1
else
    echo "All training processes completed successfully!"
    exit 0
fi