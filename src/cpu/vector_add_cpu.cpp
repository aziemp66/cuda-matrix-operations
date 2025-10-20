#include "cpu.h"

void vectorAddFloatCPU(const float *A, const float *B, float *C, int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

void vectorAddDoubleCPU(const double *A, const double *B, double *C, int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}