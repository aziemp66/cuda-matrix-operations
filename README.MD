# CUDA Matrix Operations — Optimized Implementations

Project demonstrating various CUDA implementations for vector and matrix operations, comparing CPU vs GPU performance and float vs double precision. Uses CMake with first-class CUDA support.

## Repository layout
- [CMakeLists.txt](CMakeLists.txt) — project configuration and CUDA settings
- [header/](header/)
  - [kernels.h](header/kernels.h) — GPU kernel declarations
  - [cpu.h](header/cpu.h) — CPU implementation declarations
  - [example.h](header/example.h) — Example function declarations
  - [logger.h](header/logger.h) — Performance logging utilities
- [src/](src/)
  - [kernels/](src/kernels/)
    - [matrix_add.cu](src/kernels/matrix_add.cu) — Matrix addition kernels
    - [matrix_mult_naive.cu](src/kernels/matrix_mult_naive.cu) — Naive matrix multiplication
    - [matrix_mult_tiled.cu](src/kernels/matrix_mult_tiled.cu) — Tiled matrix multiplication
    - [vector_add.cu](src/kernels/vector_add.cu) — Vector addition kernels
  - [example/](src/example/)
    - [matrix_add_example.cu](src/example/matrix_add_example.cu) — Matrix addition examples
    - [matrix_mult_naive_example.cu](src/example/matrix_mult_naive_example.cu) — Naive multiplication examples
    - [matrix_mult_tiled_example.cu](src/example/matrix_mult_tiled_example.cu) — Tiled multiplication examples
    - [vector_add_example.cu](src/example/vector_add_example.cu) — Vector addition examples
  - [cpu/](src/cpu/) — CPU implementations
- [main.cpp](main.cpp) — Entry point and benchmarks
- [build/](build/) — CMake build directory (ignored by .gitignore)

## Features
- Vector Addition (CPU/GPU, float/double)
- Matrix Addition (CPU/GPU, float/double)
- Matrix Multiplication
  - Naive Implementation (CPU/GPU, float/double)
  - Tiled Implementation using shared memory (CPU/GPU, float/double)
- Performance logging and comparison
- Comprehensive error handling for CUDA operations

## Prerequisites
- CMake >= 3.18
- CUDA Toolkit (nvcc) matching your GPU drivers
- A supported host compiler (gcc/clang) compatible with your CUDA toolkit

Verify:
```sh
nvcc --version
cmake --version
```

## Build
Create an out-of-source build and compile:

```sh
mkdir -p build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j$(nproc)
```

## Run

The program accepts several command-line arguments to customize execution. Run from the `build/` directory:

```bash
./matrix_ops [-tpb=<value>] [-size=<value>] [-path=<path>] [-task=<tasks>]
```

### Command-line Arguments

#### 1. Thread Block Size (-tpb)
- Purpose: Sets the number of threads per block (in each dimension for 2D blocks)
- Default: 32 (resulting in 32x32 thread blocks)
- Format: `-tpb=<positive_integer>`
- Examples:
  ```bash
  ./matrix_ops -tpb=16    # Use 16x16 thread blocks
  ./matrix_ops -tpb=32    # Use 32x32 thread blocks (default)
  ```

#### 2. Problem Size (-size)
- Purpose: Sets the size of vectors/matrices
- Default: 1024
- Format: `-size=<positive_integer>`
- For vectors: Determines number of elements
- For matrices: Determines dimensions (n x n)
- Examples:
  ```bash
  ./matrix_ops -size=1024  # 1024-element vectors, 1024x1024 matrices
  ./matrix_ops -size=2048  # 2048-element vectors, 2048x2048 matrices
  ```

#### 3. Log File Path (-path)
- Purpose: Specifies where to save performance results
- Default: "performance.csv"
- Format: `-path=<filepath>`
- Examples:
  ```bash
  ./matrix_ops -path=results.csv          # Save to results.csv
  ./matrix_ops -path=logs/benchmark.csv   # Save to logs/benchmark.csv
  ```

#### 4. Task Selection (-task)
- Purpose: Specifies which operations to run
- Format: `-task=<taskname>_<platform>_<type>`
- Components:
  - `taskname`: Operation to perform
    - `vectoradd`
    - `matrixadd`
    - `matrixmultnaive`
    - `matrixmulttiled`
  - `platform`: Execution platform
    - `cpu`
    - `gpu`
  - `type`: Data precision
    - `float`
    - `double`
- Multiple tasks: Use comma separation (no spaces)
- Examples:
  ```bash
  # Single task
  ./matrix_ops -task=vectoradd_gpu_float

  # Multiple tasks
  ./matrix_ops -task=matrixmulttiled_gpu_float,matrixmulttiled_cpu_float

  # Different precisions
  ./matrix_ops -task=matrixadd_gpu_float,matrixadd_gpu_double
  ```

### Complete Examples

1. Basic vector addition with custom size:
```bash
./matrix_ops -task=vectoradd_gpu_float -size=1048576
```

2. Compare CPU vs GPU matrix multiplication:
```bash
./matrix_ops -task=matrixmultnaive_cpu_float,matrixmultnaive_gpu_float -size=1024 -path=mult_comparison.csv
```

3. Benchmark different thread block sizes:
```bash
./matrix_ops -task=matrixmulttiled_gpu_float -tpb=16 -size=2048
./matrix_ops -task=matrixmulttiled_gpu_float -tpb=32 -size=2048
```

4. Full precision comparison:
```bash
./matrix_ops -task=matrixadd_gpu_float,matrixadd_gpu_double,matrixadd_cpu_float,matrixadd_cpu_double -size=2048
```

### Notes
- If no tasks are specified, all operations will be executed
- Invalid task names will trigger an error message
- TPB must be a positive integer
- Size must be a positive integer
- Results are automatically logged to the specified path
- All arguments are optional and have default values

This runs benchmarks comparing:
- CPU vs GPU implementations
- Float vs Double precision
- Different thread block sizes
- Different matrix/vector sizes

## Performance Metrics
The program measures and outputs:
- Execution time in milliseconds
- CPU vs GPU speedup ratios
- Impact of precision (float vs double)
- Effect of thread block sizes on performance

## Error Handling
- Comprehensive CUDA error checking
- Memory allocation validation
- Kernel launch verification
- Device synchronization checks
- Memory transfer validation

## Implementation Notes
- Tiled multiplication uses shared memory for better data reuse
- Vector and matrix sizes are configurable through parameters
- Thread block sizes can be tuned for optimal performance
- Results are validated between CPU and GPU implementations
