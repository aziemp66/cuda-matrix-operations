#include "cpu.h"
#include "example.h"
#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// CPU wrapper for float
void RunMatrixAddCPUFloat(const float *h_A, const float *h_B, float *h_C, int M,
                          int N) {
  clock_t start_cpu = clock();
  matrixAddCPU(h_A, h_B, h_C, M, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (float): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for float
void RunMatrixAddGPUFloat(const float *h_A, const float *h_B, float *h_C, int M,
                          int N) {
  const int numElements = M * N;
  const size_t size = numElements * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
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

// CPU wrapper for double
void RunMatrixAddCPUDouble(const double *h_A, const double *h_B, double *h_C,
                           int M, int N) {
  clock_t start_cpu = clock();
  matrixAddCPU(h_A, h_B, h_C, M, N);
  clock_t end_cpu = clock();
  float cpu_duration_ms =
      1000.0f * (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time (double): %f ms\n", cpu_duration_ms);
}

// GPU wrapper for double
void RunMatrixAddGPUDouble(const double *h_A, const double *h_B, double *h_C,
                           int M, int N) {
  const int numElements = M * N;
  const size_t size = numElements * sizeof(double);

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  clock_t start_gpu = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
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
void MatrixAddExampleFloat() {
  printf("\nInitiating Matrix Addition (Float Precision)\n");
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  float *h_C_Cpu = new float[numElements];
  float *h_C_Gpu = new float[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  RunMatrixAddCPUFloat(h_A, h_B, h_C_Cpu, M, N);
  RunMatrixAddGPUFloat(h_A, h_B, h_C_Gpu, M, N);

  double diff = 0.0;
  for (int i = 0; i < numElements; ++i) {
    diff += std::fabs((double)h_C_Cpu[i] - (double)h_C_Gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

void MatrixAddExampleDouble() {
  printf("\nInitiating Matrix Addition (Double Precision)\n");
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;

  double *h_A = new double[numElements];
  double *h_B = new double[numElements];
  double *h_C_Cpu = new double[numElements];
  double *h_C_Gpu = new double[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  RunMatrixAddCPUDouble(h_A, h_B, h_C_Cpu, M, N);
  RunMatrixAddGPUDouble(h_A, h_B, h_C_Gpu, M, N);

  double diff = 0.0;
  for (int i = 0; i < numElements; ++i) {
    diff += std::fabs(h_C_Cpu[i] - h_C_Gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", diff);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C_Cpu;
  delete[] h_C_Gpu;
}

float MatrixAddExampleCPUFloat() {
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  float *h_C = new float[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixAddCPU(h_A, h_B, h_C, M, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Add Float Time: %f ms\n", duration);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return duration;
}

float MatrixAddExampleGPUFloat() {
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;
  const size_t size = numElements * sizeof(float);

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  float *h_C = new float[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clock_t start = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Add Float Time: %f ms\n", duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return duration;
}

float MatrixAddExampleCPUDouble() {
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;

  double *h_A = new double[numElements];
  double *h_B = new double[numElements];
  double *h_C = new double[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixAddCPU(h_A, h_B, h_C, M, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Add Double Time: %f ms\n", duration);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return duration;
}

float MatrixAddExampleGPUDouble() {
  const int M = 1 << 10;
  const int N = 1 << 10;
  const int numElements = M * N;
  const size_t size = numElements * sizeof(double);

  double *h_A = new double[numElements];
  double *h_B = new double[numElements];
  double *h_C = new double[numElements];

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clock_t start = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Add Double Time: %f ms\n", duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return duration;
}