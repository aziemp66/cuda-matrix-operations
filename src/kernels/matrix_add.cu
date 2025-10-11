// File: src/kernels/matrix_add.cu

#include "kernels.h"

__global__ void matrixAdd(const float *A, const float *B, float *C, int M,
                          int N) {
  // Calculate the global row and column index for the thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check to prevent out-of-bounds memory access
  if (row < M && col < N) {
    int idx = row * N + col;
    C[idx] = A[idx] + B[idx];
  }
}