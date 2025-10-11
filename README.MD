# CUDA Matrix Operations — Naive Implementations

Small demo project with CUDA kernels for matrix addition and matrix multiplication (naive and tiled). Uses CMake with first-class CUDA support.

## Repository layout
- [CMakeLists.txt](CMakeLists.txt) — project configuration and CUDA settings.
- [header/kernels.h](header/kernels.h) — GPU kernel declarations (e.g. [`matrixAdd`](header/kernels.h), [`matMulNaiveKernel`](header/kernels.h), [`matMulTiledKernel`](header/kernels.h)).
- [src/main.cu](src/main.cu) — entry point and tests (functions: [`testMatrixAddition`](src/main.cu), [`testMatrixMult`](src/main.cu), [`testMatrixMultTiled`](src/main.cu)).
- src/kernels/
  - [src/kernels/matrix_add.cu](src/kernels/matrix_add.cu) — `matrixAdd` kernel implementation.
  - [src/kernels/matrix_mult_naive.cu](src/kernels/matrix_mult_naive.cu) — `matMulNaiveKernel`.
  - [src/kernels/matrix_mult_tiled.cu](src/kernels/matrix_mult_tiled.cu) — `matMulTiledKernel` (tiled/shared-memory).
- build/ — out-of-source CMake build directory (ignored by .gitignore).

## Prerequisites
- CMake >= 3.18
- CUDA Toolkit (nvcc) matching your GPU drivers
- A supported host compiler (gcc/clang) compatible with your CUDA toolkit

Verify:
```sh
nvcc --version
cmake --version
```

## Build (recommended)
Create an out-of-source build and compile the executable:

```sh
mkdir -p build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j$(nproc)
```

Alternative (basic):
```sh
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The target executable `matrix_ops` is produced in `build/`.

## Run
From `build/`:
```sh
./matrix_ops
```
This runs the tests implemented in [src/main.cu](src/main.cu): [`testMatrixAddition`](src/main.cu), [`testMatrixMult`](src/main.cu), [`testMatrixMultTiled`](src/main.cu).

## Notes & Troubleshooting
- CMake uses `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt) to control GPU code generation. If you see runtime JIT or missing SASS for your GPU, add the correct architecture code(s) there.
- If you get errors like "Unknown argument: '--generate-code=...'" or messages about `-forward-unknown-to-host-compiler`, it usually means nvcc flags are being parsed by a non-CUDA host compiler. Confirm CMake selected the CUDA toolchain (nvcc) and that `nvcc --version` works. See [CMakeLists.txt](CMakeLists.txt) for how CUDA is enabled.
- If linking fails, ensure CUDA runtime is available and the compiler/linker paths match your CUDA installation.

## Implementation pointers
- Tiled kernel uses shared memory for better reuse: see [src/kernels/matrix_mult_tiled.cu](src/kernels/matrix_mult_tiled.cu).
- Naive kernel is implemented in [src/kernels/matrix_mult_naive.cu](src/kernels/matrix_mult_naive.cu).
- Addition kernel in [src/kernels/matrix_add.cu](src/kernels/matrix_add.cu).
- Declarations live in [header/kernels.h](header/kernels.h).
