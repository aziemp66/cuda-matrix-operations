#include "cpu.h"
#include "example.h"
#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

void VectorAddExampleFloat() {
  printf("\nInitiating Vector Addition (Float Precision)\n");
  const int N = 1 << 20;
  const size_t size = N * sizeof(float);

  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C_Cpu = new float[N];
  float *h_C_Gpu = new float[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start_cpu = clock();
  vectorAddFloatCPU(h_A, h_B, h_C_Cpu, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  float *d_A, *d_B, *d_C;
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  clock_t start_gpu = clock();
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  vectorAddFloat<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C_Gpu, d_C, size, cudaMemcpyDeviceToHost);
  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0 * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time: %f ms\n", gpu_duration_ms);

  float diff = 0.0f;
  for (int i = 0; i < N; ++i) {
    diff += fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", diff);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

void VectorAddExampleDouble() {
  printf("\nInitiating Vector Addition (Double Precision)\n");
  const int N = 1 << 20;
  const size_t size = N * sizeof(double);

  double *h_A = new double[N];
  double *h_B = new double[N];
  double *h_C_Cpu = new double[N];
  double *h_C_Gpu = new double[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start_cpu = clock();
  vectorAddDoubleCPU(h_A, h_B, h_C_Cpu, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0 * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  double *d_A, *d_B, *d_C;
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  clock_t start_gpu = clock();
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  vectorAddDouble<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C_Gpu, d_C, size, cudaMemcpyDeviceToHost);
  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0 * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time: %f ms\n", gpu_duration_ms);

  double diff = 0.0;
  for (int i = 0; i < N; ++i) {
    diff += fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", diff);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}
