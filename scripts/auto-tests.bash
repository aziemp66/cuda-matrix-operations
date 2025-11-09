#!/bin/bash

# --- Configuration ---
# Set the path to your compiled binary
BINARY_PATH="./build/matrix_ops"

# Set the arguments for the run
# Note: The log file will be appended to on each run.
LOG_FILE="results/benchmark_run_2.csv"
TASKS="matrixmultnaive_gpu_float,matrixmulttiled_gpu_float"
NUM_RUNS=10

# Define the range of sizes (as exponents for 1 << N)
START_EXP=8
END_EXP=12

# Define the list of TPB values to test
TPB_LIST="8 16 32"

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
echo "Will run $NUM_RUNS iterations for each size from 1 << $START_EXP to 1 << $END_EXP"
echo "Will test TPB values: $TPB_LIST"
echo "  Binary: $BINARY_PATH"
echo "  Tasks:  $TASKS"
echo "  Log:    $LOG_FILE"
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
    echo "  ======================================"
    echo "  ### Testing TPB: $tpb ###"
    echo "  ======================================"
    echo ""

    # Inner loop for running 10 times per size and TPB
    for i in $(seq 1 $NUM_RUNS)
    do
      echo "--- [ Run $i of $NUM_RUNS for size $SIZE, TPB $tpb ] ---"
      
      # Execute the binary with the specified arguments
      # The program will append the results to the log file
      $BINARY_PATH -task="$TASKS" -size="$SIZE" -tpb="$tpb" -path="$LOG_FILE"
      
      echo "-------------------------------------------------"
      echo ""
      
      # Optional: Add a small delay between runs if needed
      # sleep 1
    done
    
    echo "" # Add a space between TPB benchmarks
  done

  echo "" # Add a space between size benchmarks
done

echo "Benchmark complete. Results are in $LOG_FILE"