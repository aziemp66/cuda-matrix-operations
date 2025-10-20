// File: src/kernels/matrix_mult.cu

#include "kernels.h"

// Define the tile size, which also dictates the block size.
// 16x16 is the optimal balance for the GTX 1650's Turing architecture.
#define TILE_SIZE 16

// float version

__global__ void matrixMultTiled(const float *A, const float *B, float *C, int M,
                                int N, int K) {
  // CORRECTED: Declare shared memory as 2D arrays to hold the tiles.
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float value = 0.0f;

  for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

    int a_row = row;
    int a_col = phase * TILE_SIZE + tx;
    if (a_row < M && a_col < K) {
      tile_A[ty][tx] = A[a_row * K + a_col];
    } else {
      tile_A[ty][tx] = 0.0f;
    }

    int b_row = phase * TILE_SIZE + ty;
    int b_col = col;
    if (b_row < K && b_col < N) {
      tile_B[ty][tx] = B[b_row * N + b_col];
    } else {
      tile_B[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      value += tile_A[ty][i] * tile_B[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}

// double version

__global__ void matrixMultTiled(const double *A, const double *B, double *C,
                                int M, int N, int K) {
  // CORRECTED: Declare shared memory as 2D arrays to hold the tiles.
  __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  double value = 0.0;

  for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

    int a_row = row;
    int a_col = phase * TILE_SIZE + tx;
    if (a_row < M && a_col < K) {
      tile_A[ty][tx] = A[a_row * K + a_col];
    } else {
      tile_A[ty][tx] = 0.0;
    }

    int b_row = phase * TILE_SIZE + ty;
    int b_col = col;
    if (b_row < K && b_col < N) {
      tile_B[ty][tx] = B[b_row * N + b_col];
    } else {
      tile_B[ty][tx] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      value += tile_A[ty][i] * tile_B[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}