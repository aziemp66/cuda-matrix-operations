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
From `build/`:
```sh
./matrix_ops
```

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
