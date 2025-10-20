#include "cpu.h"

// float version
void matrixAddCPU(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      C[idx] = A[idx] + B[idx];
    }
  }
}

// double version
void matrixAddCPU(const double *A, const double *B, double *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      C[idx] = A[idx] + B[idx];
    }
  }
}