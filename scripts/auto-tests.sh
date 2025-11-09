#!/bin/bash

# --- Default Configuration ---
# Set the path to your compiled binary
BINARY_PATH="./build/matrix_ops"

# Default run arguments
LOG_FILE="results/benchmark_run.csv"
TASKS="matrixmultnaive_gpu_float,matrixmulttiled_gpu_float"
NUM_RUNS=10

# Default Benchmark Loops
START_EXP=8
END_EXP=12
TPB_LIST="8 16 32"

# --- Argument Parsing ---
# Parses arguments in the format -key=value, similar to the C++ program
for arg in "$@"
do
    case $arg in
        -task=*)
        TASKS="${arg#*=}"
        shift
        ;;
        -path=*)
        LOG_FILE="${arg#*=}"
        shift
        ;;
        -iterations=*)
        NUM_RUNS="${arg#*=}"
        shift
        ;;
        -tpb=*)
        # Accepts a space-separated list for the benchmark loop (e.g., -tpb="8 16 32")
        TPB_LIST="${arg#*=}"
        shift
        ;;
        -exp_range=*)
        # Accepts range in format START:END (e.g., -exp_range=8:12)
        RANGE="${arg#*=}"
        START_EXP=$(echo "$RANGE" | cut -d':' -f1)
        END_EXP=$(echo "$RANGE" | cut -d':' -f2)
        shift
        ;;
        -h|--help)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -task=<list>       Comma-separated list of tasks (default: $TASKS)"
        echo "  -path=<file>       Path to log file (default: $LOG_FILE)"
        echo "  -iterations=<int>  Number of runs per configuration (default: $NUM_RUNS)"
        echo "  -tpb=<list>        Space-separated list of TPB values to test (default: \"$TPB_LIST\")"
        echo "  -exp_range=<S:E>   Start and End exponents for size (1<<N) (default: $START_EXP:$END_EXP)"
        exit 0
        ;;
        *)
        # Ignore unknown arguments or pass them through if needed
        shift
        ;;
    esac
done

# --- Check if binary exists ---
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found at $BINARY_PATH"
    echo "Please build the project first."
    exit 1
fi

# --- Create results directory if it doesn't exist ---
mkdir -p "$(dirname "$LOG_FILE")"

# --- Main Loop ---
echo "Starting benchmark..."
echo "Configuration:"
echo "  Binary:      $BINARY_PATH"
echo "  Tasks:       $TASKS"
echo "  Log File:    $LOG_FILE"
echo "  Iterations:  $NUM_RUNS"
echo "  Size Range:  1<<$START_EXP to 1<<$END_EXP"
echo "  TPB Values:  $TPB_LIST"
echo ""

# Outer loop for iterating through sizes
for size_exp in $(seq $START_EXP $END_EXP)
do
    SIZE="1 << $size_exp"
    # This shell arithmetic evaluates the size for logging purposes
    SIZE_EVAL=$((1 << $size_exp))
    
    echo "################################################"
    echo "### Testing Size: $SIZE (evaluates to $SIZE_EVAL) ###"
    echo "################################################"
    echo ""

    # Middle loop for iterating through TPB values
    for tpb in $TPB_LIST
    do
        echo "   ======================================"
        echo "   ### Testing TPB: $tpb ###"
        echo "   ======================================"
        echo ""

        # Inner loop for running N iterations per size and TPB
        for i in $(seq 1 $NUM_RUNS)
        do
            echo "--- [ Run $i of $NUM_RUNS for size $SIZE, TPB $tpb ] ---"
            
            # Execute the binary with the specified arguments
            $BINARY_PATH -task="$TASKS" -size="$SIZE" -tpb="$tpb" -path="$LOG_FILE"
            
            echo "-------------------------------------------------"
            echo ""
        done
        
        echo "" # Add a space between TPB benchmarks
    done

    echo "" # Add a space between size benchmarks
done

echo "Benchmark complete. Results are in $LOG_FILE"