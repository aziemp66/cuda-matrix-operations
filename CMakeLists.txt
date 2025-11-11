# File: CMakeLists.txt

# Specify the minimum required version of CMake.
# Version 3.18 or newer is highly recommended for the best CUDA support.
cmake_minimum_required(VERSION 3.18)

# Define the project name and specify that it uses both C++ (CXX) and CUDA.
# This is the key to enabling first-class language support.
project(CudaMatrixOps LANGUAGES CXX CUDA)

# Find the installed CUDA Toolkit on the system.
find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the C++ standard for the entire project.
# This applies to both.cpp and.cu files compiled by the host compiler.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the C++ standard for CUDA compilation (for device code).
# This ensures nvcc uses the correct C++ dialect.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for LSP (Language Server Protocol) support
# This generates compile_commands.json that tools like clangd, ccls, etc. can use
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify the target GPU architectures to generate native SASS code.
# This avoids runtime JIT compilation and improves performance.
# '75' for Turing, '80' for Ampere. Add or remove based on your GPU.
set(CMAKE_CUDA_ARCHITECTURES 75 80)

# Add the executable target. CMake will automatically use the C++ compiler for.cpp
# files and the CUDA compiler (nvcc) for.cu files.
add_executable(
    matrix_ops
    main.cpp
    src/config/config.cpp
    src/controller/vector_add_controller.cu
    src/controller/matrix_add_controller.cu
    src/controller/matrix_mult_naive_controller.cu
    src/controller/matrix_mult_tiled_controller.cu
    src/cpu/vector_add_cpu.cpp
    src/cpu/matrix_add_cpu.cpp
    src/cpu/matrix_mult_cpu.cpp
    src/kernels/vector_add.cu
    src/kernels/matrix_add.cu
    src/kernels/matrix_mult_naive.cu
    src/kernels/matrix_mult_tiled.cu
    src/logger/logger.cpp
    src/task_executor/task_executor.cpp
    src/arg_parser/arg_parser.cpp
)

# Modern CMake creates imported targets for CUDA libraries.
# To link against the CUDA runtime library, use the CUDA::cudart target.
# This handles all necessary linker flags and paths automatically.
target_link_libraries(matrix_ops PRIVATE CUDA::cudart)

# Tell the compiler where to find our header files.
target_include_directories(matrix_ops PRIVATE header)