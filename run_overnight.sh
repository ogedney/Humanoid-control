#!/bin/bash

# Instructions:
# To run this script, type ./run_overnight.sh in the terminal
# To check progress, type tail -f logs/training_log_*.txt in the terminal

# Create logs directory if it doesn't exist
mkdir -p logs

# Prevent Mac from sleeping while script is running
caffeinate -i -m -s &
CAFFEINATE_PID=$!

# Log file for training output
LOG_FILE="logs/training_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Starting training at $(date)" > $LOG_FILE

# Function to handle script termination
cleanup() {
    echo "Stopping training at $(date)" >> $LOG_FILE
    # Kill the caffeinate process to allow Mac to sleep again
    kill $CAFFEINATE_PID
    exit 0
}

# Set up trap to handle termination signals
trap cleanup SIGINT SIGTERM EXIT

# Run the training in no-GUI mode
echo "Running training with output logged to $LOG_FILE" 
python main.py --no-gui >> $LOG_FILE 2>&1

# Script will exit via cleanup function
