#include "kernels.h"

#define TILE_SIZE 16 // baseline tile size; can be tuned to 8, 16, 32

// -------------------------------------------------------------------------------------------------
// Optimized Tiled Matrix Multiplication Kernel (float)
// Implements: shared memory tiling, unrolling, restrict, coalesced access, and
// fast-math support. References in comments refer to NVIDIA CUDA documentation
// or official blogs.
// -------------------------------------------------------------------------------------------------

// Using __restrict__ and const reduce instructions
// (CUDA C Programming Guide [164])
__global__ void
matrixMultTiled(const float *__restrict__ A, // __restrict__ avoids aliasing
                const float *__restrict__ B, float *__restrict__ C, int M,
                int N, int K) {
  // Shared memory tiles — each block computes a TILE_SIZE×TILE_SIZE submatrix
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float value = 0.0f;

  // Loop over all tiles (phases)
  // (CUDA C Programming Guide [275])
  for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {
    int a_col = phase * TILE_SIZE + tx;
    int b_row = phase * TILE_SIZE + ty;

    // Coalesced global memory access pattern (Best Practices §3.2.5)
    tile_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    tile_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

// Unrolled inner loop (CUDA C Programming Guide §5.3.2)
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      value += tile_A[ty][i] * tile_B[i][tx];
    }

    __syncthreads();
  }

  // Write result back to global memory
  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}

// -------------------------------------------------------------------------------------------------
// Optimized Tiled Matrix Multiplication Kernel (double)
// Note: Fast math is not used for double precision (IEEE compliance).
// -------------------------------------------------------------------------------------------------

__global__ void matrixMultTiled(const double *__restrict__ A,
                                const double *__restrict__ B,
                                double *__restrict__ C, int M, int N, int K) {
  __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  double value = 0.0;

  for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {
    int a_col = phase * TILE_SIZE + tx;
    int b_row = phase * TILE_SIZE + ty;

    tile_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
    tile_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      value += tile_A[ty][i] * tile_B[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}
