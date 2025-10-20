#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// --- Kernel Declarations ---
// These are the functions that will run on the GPU.
// They are defined in the.cu files.

// vector addition
__global__ void vectorAddFloat(const float *A, const float *B, float *C, int N);

__global__ void vectorAddDouble(const double *A, const double *B, double *C,
                                int N);

// float versions
__global__ void matrixAdd(const float *A, const float *B, float *C, int M,
                          int N);

__global__ void matrixMultNaive(const float *A, const float *B, float *C, int M,
                                int N, int K);

__global__ void matrixMultTiled(const float *A, const float *B, float *C, int M,
                                int N, int K);

// double versions
__global__ void matrixAdd(const double *A, const double *B, double *C, int M,
                          int N);

__global__ void matrixMultNaive(const double *A, const double *B, double *C,
                                int M, int N, int K);

__global__ void matrixMultTiled(const double *A, const double *B, double *C,
                                int M, int N, int K);

#endif // KERNELS_H