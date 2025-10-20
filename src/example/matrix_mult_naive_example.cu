#include "cpu.h"
#include "example.h"
#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

void MatrixMultNaiveExampleFloat() {
  printf("\nInitiating Matrix Naive Multiplication (Float Precision)\n");
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int K = 1 << 10;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;
  const size_t sizeA = numElementsA * sizeof(float);
  const size_t sizeB = numElementsB * sizeof(float);
  const size_t sizeC = numElementsC * sizeof(float);

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

  clock_t start_cpu = clock();
  matrixMultCPU(h_A, h_B, h_C_Cpu, M, N, K);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  float *d_A, *d_B, *d_C;
  clock_t start_gpu = clock();
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);
  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

  cudaDeviceSynchronize();
  cudaMemcpy(h_C_Gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0 * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time: %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  float diff = 0.0f;
  for (int i = 0; i < numElementsC; ++i) {
    diff += fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

void MatrixMultNaiveExampleDouble() {
  printf("\nInitiating Matrix Naive Multiplication (Double Precision)\n");
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int K = 1 << 10;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;
  const size_t sizeA = numElementsA * sizeof(double);
  const size_t sizeB = numElementsB * sizeof(double);
  const size_t sizeC = numElementsC * sizeof(double);

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

  clock_t start_cpu = clock();
  matrixMultCPU(h_A, h_B, h_C_Cpu, M, N, K);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double *d_A, *d_B, *d_C;
  clock_t start_gpu = clock();
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);
  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

  cudaDeviceSynchronize();
  cudaMemcpy(h_C_Gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0 * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time: %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  double diff = 0.0;
  for (int i = 0; i < numElementsC; ++i) {
    diff += fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  }

  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}