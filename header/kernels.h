#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// --- Kernel Declarations ---
// These are the functions that will run on the GPU.
// They are defined in the.cu files.

__global__ void matrixAdd(const float *A, const float *B, float *C, int M,
                          int N);

__global__ void matMulNaiveKernel(const float *A, const float *B, float *C,
                                  int M, int N, int K);

__global__ void matMulTiledKernel(const float *A, const float *B, float *C,
                                  int M, int N, int K);

__global__ void scaleMatrix(float *matrix, int N);

#endif // KERNELS_H