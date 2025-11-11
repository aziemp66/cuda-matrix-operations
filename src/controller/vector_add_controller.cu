#include "controller.h"
#include "cpu.h"
#include "kernels.h"
#include "logger.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

float VectorAddControllerCPU(std::vector<float> &h_C, int n) {
  const int N = n;
  float *h_A = new float[N];
  float *h_B = new float[N];
  h_C.resize(N);

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  vectorAddFloatCPU(h_A, h_B, h_C.data(), N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Vector Add Float Time: %f ms (Size: %d)\n", duration, N);

  logResult(TaskType::VECTOR_ADD, Platform::CPU, N, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float VectorAddControllerGPU(std::vector<float> &h_C, int TPB, int n) {
  const int N = n;
  float *h_A = new float[N];
  float *h_B = new float[N];
  h_C.resize(N);

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  const size_t size = N * sizeof(float);

  clock_t start = clock();

  // Allocate GPU memory with error checking
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
    printf("CUDA malloc failed for d_A: %s\n", cudaGetErrorString(err));
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess) {
    printf("CUDA malloc failed for d_B: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess) {
    printf("CUDA malloc failed for d_C: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  // Copy data to device with error checking
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA memcpy failed for d_A: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA memcpy failed for d_B: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  dim3 threadsPerBlock(TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("CUDA memcpy failed for h_C: %s\n", cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    return -1.0f;
  }

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Vector Add Float Time: %f ms (TPB: %d, Size: %d)\n", duration,
         TPB, N);

  logResult(TaskType::VECTOR_ADD, Platform::GPU, TPB, N, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}

float VectorAddControllerCPU(std::vector<double> &h_C, int n) {
  const int N = n;
  double *h_A = new double[N];
  double *h_B = new double[N];
  h_C.resize(N);

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  vectorAddDoubleCPU(h_A, h_B, h_C.data(), N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Vector Add Double Time: %f ms (Size: %d)\n", duration, N);

  logResult(TaskType::VECTOR_ADD, Platform::CPU, N, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float VectorAddControllerGPU(std::vector<double> &h_C, int TPB, int n) {
  const int N = n;
  double *h_A = new double[N];
  double *h_B = new double[N];
  h_C.resize(N);

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

  cudaDeviceSynchronize();

  cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Vector Add Double Time: %f ms (TPB: %d, Size: %d)\n", duration,
         TPB, N);

  logResult(TaskType::VECTOR_ADD, Platform::GPU, TPB, N, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}
