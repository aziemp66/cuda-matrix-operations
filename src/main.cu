// File: src/main.cpp

#include "kernels.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// --- CPU Implementations (updated to use float) ---
void matrixAddCPU(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      C[idx] = A[idx] + B[idx];
    }
  }
}

void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// --- Test Functions ---
void testMatrixAddition() {
  printf("\n--- Testing Naive Matrix Addition (float) ---\n");
  int M = 1024, N = 1024;
  int numElements = M * N;
  size_t size = numElements * sizeof(float);

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  float *h_C_gpu = new float[numElements];
  float *h_C_cpu = new float[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = (float)rand() / RAND_MAX;
    h_B[i] = (float)rand() / RAND_MAX;
  }

  clock_t start_cpu = clock();
  matrixAddCPU(h_A, h_B, h_C_cpu, M, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);
  matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
  cudaEventRecord(stop_gpu);

  cudaEventSynchronize(stop_gpu);
  float gpu_duration_ms = 0;
  cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu);
  printf("GPU kernel execution time: %f ms\n", gpu_duration_ms);

  cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

  float error = 0.0;
  for (int i = 0; i < numElements; ++i) {
    error += fabs(h_C_cpu[i] - h_C_gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", error);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_cpu;
  delete[] h_C_gpu;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
}

void testMatrixMult() {
  printf("\n--- Testing Naive Matrix Multiplication (float) ---\n");
  int M = 512, N = 512, K = 512;
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C_gpu = new float[M * N];
  float *h_C_cpu = new float[M * N];

  for (int i = 0; i < M * K; ++i)
    h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < K * N; ++i)
    h_B[i] = (float)rand() / RAND_MAX;

  clock_t start_cpu = clock();
  matMulCPU(h_A, h_B, h_C_cpu, M, N, K);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);
  matMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_duration_ms = 0;
  cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu);
  printf("GPU Naive Kernel execution time: %f ms\n", gpu_duration_ms);

  cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

  float error = 0.0;
  for (int i = 0; i < M * N; ++i) {
    error += fabs(h_C_cpu[i] - h_C_gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", error);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_cpu;
  delete[] h_C_gpu;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
}

void testMatrixMultTiled() {
  printf("\n--- Testing Tiled Matrix Multiplication (float) ---\n");
  int M = 512, N = 512, K = 512;
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C_gpu = new float[M * N];
  float *h_C_cpu = new float[M * N];

  for (int i = 0; i < M * K; ++i)
    h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < K * N; ++i)
    h_B[i] = (float)rand() / RAND_MAX;

  // We still run the CPU version for verification
  matMulCPU(h_A, h_B, h_C_cpu, M, N, K);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  // Use the optimal TILE_SIZE for the launch configuration
  const int TILE_SIZE = 16;
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);
  matMulTiledKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_duration_ms = 0;
  cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu);
  printf("GPU Tiled Kernel execution time: %f ms\n", gpu_duration_ms);

  cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

  float error = 0.0;
  for (int i = 0; i < M * N; ++i) {
    error += fabs(h_C_cpu[i] - h_C_gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", error);

  delete h_A;
  delete h_B;
  delete h_C_cpu;
  delete h_C_gpu;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
}

int main() {
  srand(time(0));
  testMatrixAddition();
  testMatrixMult();
  testMatrixMultTiled();
  return 0;
}