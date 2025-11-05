#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>

// Helper macro to check CUDA errors (returns error code)
#define CHECK_CUDA_ERROR(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
  } while(0)

// Helper macro to check CUDA malloc with error handling
#define CHECK_CUDA_MALLOC(ptr, size) \
  do { \
    cudaError_t err = cudaMalloc((void**)&(ptr), size); \
    if (err != cudaSuccess) { \
      printf("CUDA malloc failed for %s: %s\n", #ptr, cudaGetErrorString(err)); \
    } \
  } while(0)

// Cleanup helper for GPU memory (safely frees pointers)
inline void cleanupGPU(float* d_A, float* d_B, float* d_C) {
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  if (d_C) cudaFree(d_C);
}

inline void cleanupGPU(double* d_A, double* d_B, double* d_C) {
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  if (d_C) cudaFree(d_C);
}

#endif // CUDA_UTILS_H
