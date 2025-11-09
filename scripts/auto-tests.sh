#!/bin/bash

# --- Default Configuration ---
BINARY_PATH="./build/matrix_ops"

# Default run arguments
CPU_LOG_FILE="results/cpu_benchmark.csv"
GPU_LOG_FILE="results/gpu_benchmark.csv"
TASKS="matrixmultnaive_gpu_float,matrixmulttiled_gpu_float,matrixmultnaive_cpu_float"
NUM_RUNS=10

# Default Benchmark Loops
START_EXP=8
END_EXP=10
TPB_LIST="8 16 32"

# --- Argument Parsing ---
for arg in "$@"
do
    case $arg in
        -task=*)
        TASKS="${arg#*=}"
        shift
        ;;
        -cpu_path=*)
        CPU_LOG_FILE="${arg#*=}"
        shift
        ;;
        -gpu_path=*)
        GPU_LOG_FILE="${arg#*=}"
        shift
        ;;
        -iterations=*)
        NUM_RUNS="${arg#*=}"
        shift
        ;;
        -tpb=*)
        TPB_LIST="${arg#*=}"
        shift
        ;;
        -exp_range=*)
        RANGE="${arg#*=}"
        START_EXP=$(echo "$RANGE" | cut -d':' -f1)
        END_EXP=$(echo "$RANGE" | cut -d':' -f2)
        shift
        ;;
        -h|--help)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -task=<list>       Comma-separated list of tasks (default: $TASKS)"
        echo "  -cpu_path=<file>   Path to CPU log file (default: $CPU_LOG_FILE)"
        echo "  -gpu_path=<file>   Path to GPU log file (default: $GPU_LOG_FILE)"
        echo "  -iterations=<int>   Number of runs per configuration (default: $NUM_RUNS)"
        echo "  -tpb=<list>        Space-separated list of TPB values (default: \"$TPB_LIST\")"
        echo "  -exp_range=<S:E>   Size exponents range (default: $START_EXP:$END_EXP)"
        exit 0
        ;;
        *)
        shift
        ;;
    esac
done

# --- Check binary ---
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found at $BINARY_PATH"
    echo "Please build the project first."
    exit 1
fi

# --- Create results directories ---
mkdir -p "$(dirname "$CPU_LOG_FILE")"
mkdir -p "$(dirname "$GPU_LOG_FILE")"

# Separate CPU and GPU tasks
CPU_TASKS=""
GPU_TASKS=""
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
    if [[ $task == *"cpu"* ]]; then
        CPU_TASKS="${CPU_TASKS:+$CPU_TASKS,}$task"
    else
        GPU_TASKS="${GPU_TASKS:+$GPU_TASKS,}$task"
    fi
done

# --- Main Loop ---
echo "Starting benchmark..."
echo "Configuration:"
echo "  Binary:      $BINARY_PATH"
echo "  CPU Tasks:   $CPU_TASKS"
echo "  GPU Tasks:   $GPU_TASKS"
echo "  CPU Log:     $CPU_LOG_FILE"
echo "  GPU Log:     $GPU_LOG_FILE"
echo "  Iterations:  $NUM_RUNS"
echo "  Size Range:  1<<$START_EXP to 1<<$END_EXP"
echo "  TPB Values:  $TPB_LIST"
echo ""

for size_exp in $(seq $START_EXP $END_EXP)
do
    SIZE="1 << $size_exp"
    SIZE_EVAL=$((1 << $size_exp))
    
    echo "################################################"
    echo "### Testing Size: $SIZE (evaluates to $SIZE_EVAL) ###"
    echo "################################################"
    echo ""

    # Run CPU tasks (only size loop)
    if [ ! -z "$CPU_TASKS" ]; then
        echo "=== Running CPU Tasks ==="
        for i in $(seq 1 $NUM_RUNS)
        do
            echo "--- [ CPU Run $i of $NUM_RUNS for size $SIZE ] ---"
            $BINARY_PATH -task="$CPU_TASKS" -size="$SIZE" \
                        -cpu_path="$CPU_LOG_FILE" -gpu_path="$GPU_LOG_FILE"
            echo "-------------------------------------------------"
            echo ""
        done
    fi

    # Run GPU tasks (size and TPB loops)
    if [ ! -z "$GPU_TASKS" ]; then
        echo "=== Running GPU Tasks ==="
        for tpb in $TPB_LIST
        do
            echo "   ======================================"
            echo "   ### Testing TPB: $tpb ###"
            echo "   ======================================"
            echo ""

            for i in $(seq 1 $NUM_RUNS)
            do
                echo "--- [ GPU Run $i of $NUM_RUNS for size $SIZE, TPB $tpb ] ---"
                $BINARY_PATH -task="$GPU_TASKS" -size="$SIZE" -tpb="$tpb" \
                            -cpu_path="$CPU_LOG_FILE" -gpu_path="$GPU_LOG_FILE"
                echo "-------------------------------------------------"
                echo ""
            done
            echo ""
        done
    fi

    echo ""
done

echo "Benchmark complete."
echo "CPU results in $CPU_LOG_FILE"
echo "GPU results in $GPU_LOG_FILE"