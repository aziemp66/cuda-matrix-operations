#include "cpu.h"
#include "example.h"
#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// CPU wrapper for float naive matmul
void RunMatrixMultNaiveCPUFloat(const float *h_A, const float *h_B, float *h_C,
                                int M, int N, int K) {
  clock_t start_cpu = clock();
  matrixMultCPU(h_A, h_B, h_C, M, N, K);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (float): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for float naive matmul
void RunMatrixMultNaiveGPUFloat(const float *h_A, const float *h_B, float *h_C,
                                int M, int N, int K) {
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;
  const size_t sizeA = numElementsA * sizeof(float);
  const size_t sizeB = numElementsB * sizeof(float);
  const size_t sizeC = numElementsC * sizeof(float);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0f * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time (float): %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// CPU wrapper for double naive matmul
void RunMatrixMultNaiveCPUDouble(const double *h_A, const double *h_B,
                                 double *h_C, int M, int N, int K) {
  clock_t start_cpu = clock();
  matrixMultCPU(h_A, h_B, h_C, M, N, K);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (double): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for double naive matmul
void RunMatrixMultNaiveGPUDouble(const double *h_A, const double *h_B,
                                 double *h_C, int M, int N, int K) {
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;
  const size_t sizeA = numElementsA * sizeof(double);
  const size_t sizeB = numElementsB * sizeof(double);
  const size_t sizeC = numElementsC * sizeof(double);

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0f * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time (double): %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// Example functions use the separated wrappers
void MatrixMultNaiveExampleFloat() {
  printf("\nInitiating Matrix Naive Multiplication (Float Precision)\n");
  const int M = 1 << 14;
  const int N = 1 << 14;
  const int K = 1 << 14;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  float *h_A = new float[numElementsA];
  float *h_B = new float[numElementsB];
  float *h_C_Cpu = new float[numElementsC];
  float *h_C_Gpu = new float[numElementsC];

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // RunMatrixMultNaiveCPUFloat(h_A, h_B, h_C_Cpu, M, N, K);
  RunMatrixMultNaiveGPUFloat(h_A, h_B, h_C_Gpu, M, N, K);

  // float diff = 0.0f;
  // for (int i = 0; i < numElementsC; ++i) {
  //   diff += std::fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  // }
  // printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

void MatrixMultNaiveExampleDouble() {
  printf("\nInitiating Matrix Naive Multiplication (Double Precision)\n");
  const int M = 1 << 14;
  const int N = 1 << 14;
  const int K = 1 << 14;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  double *h_A = new double[numElementsA];
  double *h_B = new double[numElementsB];
  double *h_C_Cpu = new double[numElementsC];
  double *h_C_Gpu = new double[numElementsC];

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  // RunMatrixMultNaiveCPUDouble(h_A, h_B, h_C_Cpu, M, N, K);
  RunMatrixMultNaiveGPUDouble(h_A, h_B, h_C_Gpu, M, N, K);

  // double diff = 0.0;
  // for (int i = 0; i < numElementsC; ++i) {
  //   diff += std::fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  // }
  // printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}