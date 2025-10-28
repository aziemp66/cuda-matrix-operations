#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// --- Kernel Declarations ---
// These are the functions that will run on the GPU.
// They are defined in the.cu files.

__global__ void vectorAdd(const float *A, const float *B, float *C, int N);

__global__ void vectorAdd(const double *A, const double *B, double *C, int N);

__global__ void matrixAdd(const float *A, const float *B, float *C, int M,
                          int N);

__global__ void matrixAdd(const double *A, const double *B, double *C, int M,
                          int N);

__global__ void matrixMultNaive(const float *A, const float *B, float *C, int M,
                                int N, int K);

__global__ void matrixMultNaive(const double *A, const double *B, double *C,
                                int M, int N, int K);

__global__ void matrixMultTiled(const float *A, const float *B, float *C, int M,
                                int N, int K);

__global__ void matrixMultTiled(const double *A, const double *B, double *C,
                                int M, int N, int K);

#endif // KERNELS_H