#include "kernels.h"

// Naive CUDA Kernel for Matrix Multiplication (C = A * B)
// A: M x K, B: K x N, C: M x N

// float version

__global__ void matrixMultNaive(const float *A, const float *B, float *C, int M,
                                int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int i = 0; i < K; ++i) {
      value += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = value;
  }
}

// double version

__global__ void matrixMultNaive(const double *A, const double *B, double *C,
                                int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double value = 0.0;
    for (int i = 0; i < K; ++i) {
      value += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = value;
  }
}