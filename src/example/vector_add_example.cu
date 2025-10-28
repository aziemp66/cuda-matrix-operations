#include "cpu.h"
#include "example.h"
#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// CPU wrapper for float vector add
void RunVectorAddCPUFloat(const float *h_A, const float *h_B, float *h_C,
                          int N) {
  clock_t start_cpu = clock();
  vectorAddFloatCPU(h_A, h_B, h_C, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (float): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for float vector add
void RunVectorAddGPUFloat(const float *h_A, const float *h_B, float *h_C,
                          int N) {
  const size_t size = N * sizeof(float);
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0f * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time (float): %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// CPU wrapper for double vector add
void RunVectorAddCPUDouble(const double *h_A, const double *h_B, double *h_C,
                           int N) {
  clock_t start_cpu = clock();
  vectorAddDoubleCPU(h_A, h_B, h_C, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (double): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for double vector add
void RunVectorAddGPUDouble(const double *h_A, const double *h_B, double *h_C,
                           int N) {
  const size_t size = N * sizeof(double);
  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end_gpu = clock();
  float gpu_duration_ms =
      1000.0f * (float)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
  printf("GPU execution time (double): %f ms\n", gpu_duration_ms);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// Example functions use the separated wrappers
void VectorAddExampleFloat() {
  printf("\nInitiating Vector Addition (Float Precision)\n");
  const int N = 1 << 25;

  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C_Cpu = new float[N];
  float *h_C_Gpu = new float[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  RunVectorAddCPUFloat(h_A, h_B, h_C_Cpu, N);
  RunVectorAddGPUFloat(h_A, h_B, h_C_Gpu, N);

  float diff = 0.0f;
  for (int i = 0; i < N; ++i)
    diff += std::fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

void VectorAddExampleDouble() {
  printf("\nInitiating Vector Addition (Double Precision)\n");
  const int N = 1 << 25;

  double *h_A = new double[N];
  double *h_B = new double[N];
  double *h_C_Cpu = new double[N];
  double *h_C_Gpu = new double[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  RunVectorAddCPUDouble(h_A, h_B, h_C_Cpu, N);
  RunVectorAddGPUDouble(h_A, h_B, h_C_Gpu, N);

  double diff = 0.0;
  for (int i = 0; i < N; ++i)
    diff += std::fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}
