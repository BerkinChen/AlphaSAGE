#!/bin/bash

# Grid search runner script
# Usage: ./run_grid_search.sh [cuda_device] [mode] [force]
#   cuda_device: CUDA device to use (default: 0)
#   mode: quick/focused/full (default: full)
#   force: force to rerun existing results (default: false)

CUDA_DEVICE=${1:-0}
MODE=${2:-full}
FORCE=${3:-false}

# Build command flags
FLAGS="--cuda $CUDA_DEVICE"

case "$MODE" in
    "quick")
        FLAGS="$FLAGS --quick"
        echo "Running quick test mode with fewer combinations and episodes"
        ;;
    "focused")
        FLAGS="$FLAGS --focused"
        echo "Running focused search with custom parameter grid"
        ;;
    "full")
        echo "Running full grid search"
        ;;
    *)
        echo "Unknown mode: $MODE. Using full mode."
        ;;
esac

if [ "$FORCE" = "force" ] || [ "$FORCE" = "true" ]; then
    FLAGS="$FLAGS --force"
    echo "Force mode enabled - will rerun existing results"
fi

echo "Using CUDA device: $CUDA_DEVICE"
echo "Starting grid search at $(date)"
echo "Command flags: $FLAGS"

# Make sure the script is executable
chmod +x grid_search.py

# Run the grid search with unbuffered output
pdm run python -u grid_search.py $FLAGS

echo "Grid search completed at $(date)"
