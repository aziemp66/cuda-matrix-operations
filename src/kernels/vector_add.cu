#include "kernels.h"

__global__ void vectorAddFloat(const float *A, const float *B, float *C,
                               int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

__global__ void vectorAddDouble(const double *A, const double *B, double *C,
                                int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}