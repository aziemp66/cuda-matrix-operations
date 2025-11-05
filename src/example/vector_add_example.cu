#include "cpu.h"
#include "example.h"
#include "kernels.h"
#include "logger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

float VectorAddExampleCPUFloat(int TPB, int n) {
  const int N = n;
  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  vectorAddFloatCPU(h_A, h_B, h_C, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Vector Add Float Time: %f ms (TPB: %d, Size: %d)\n", duration, TPB, N);
  
  logResult(TaskType::VECTOR_ADD, Platform::CPU, TPB, N, duration);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return duration;
}

float VectorAddExampleGPUFloat(int TPB, int n) {
  const int N = n;
  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  const size_t size = N * sizeof(float);

  clock_t start = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Vector Add Float Time: %f ms (TPB: %d, Size: %d)\n", duration, TPB, N);
  
  logResult(TaskType::VECTOR_ADD, Platform::GPU, TPB, N, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return duration;
}

float VectorAddExampleCPUDouble(int TPB, int n) {
  const int N = n;
  double *h_A = new double[N];
  double *h_B = new double[N];
  double *h_C = new double[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  vectorAddDoubleCPU(h_A, h_B, h_C, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Vector Add Double Time: %f ms (TPB: %d, Size: %d)\n", duration, TPB, N);
  
  logResult(TaskType::VECTOR_ADD, Platform::CPU, TPB, N, duration);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return duration;
}

float VectorAddExampleGPUDouble(int TPB, int n) {
  const int N = n;
  double *h_A = new double[N];
  double *h_B = new double[N];
  double *h_C = new double[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  const size_t size = N * sizeof(double);

  clock_t start = clock();

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Vector Add Double Time: %f ms (TPB: %d, Size: %d)\n", duration, TPB, N);
  
  logResult(TaskType::VECTOR_ADD, Platform::GPU, TPB, N, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return duration;
}
