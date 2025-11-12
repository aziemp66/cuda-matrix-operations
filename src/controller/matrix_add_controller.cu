#include "controller.h"
#include "cpu.h"
#include "kernels.h"
#include "logger.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

float MatrixAddControllerCPUFloat(std::vector<float> &h_C, int n) {
  const int M = n;
  const int N = n;
  const int numElements = M * N;

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  h_C.resize(numElements);

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixAddCPU(h_A, h_B, h_C.data(), M, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Add Float Time: %f ms (Size: %d x %d)\n", duration, M, N);

  logResult(TaskType::MATRIX_ADD, Platform::CPU, M, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float MatrixAddControllerGPUFloat(std::vector<float> &h_C, int TPB, int n) {
  const int M = n;
  const int N = n;
  const int numElements = M * N;
  const size_t size = numElements * sizeof(float);

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  h_C.resize(numElements);

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

  dim3 threadsPerBlock(TPB, TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Add Float Time: %f ms (TPB: %d, Size: %d x %d)\n",
         duration, TPB * TPB, M, N);

  logResult(TaskType::MATRIX_ADD, Platform::GPU, TPB * TPB, M, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}

float MatrixAddControllerCPUDouble(std::vector<double> &h_C, int n) {
  const int M = n;
  const int N = n;
  const int numElements = M * N;

  double *h_A = new double[numElements];
  double *h_B = new double[numElements];
  h_C.resize(numElements);

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixAddCPU(h_A, h_B, h_C.data(), M, N);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Add Double Time: %f ms ( Size: %d x %d)\n", duration, M,
         N);

  logResult(TaskType::MATRIX_ADD, Platform::CPU, M, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float MatrixAddControllerGPUDouble(std::vector<double> &h_C, int TPB, int n) {
  const int M = n;
  const int N = n;
  const int numElements = M * N;
  const size_t size = numElements * sizeof(double);

  double *h_A = new double[numElements];
  double *h_B = new double[numElements];
  h_C.resize(numElements);

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

  dim3 threadsPerBlock(TPB, TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Add Double Time: %f ms (TPB: %d, Size: %d x %d)\n",
         duration, TPB * TPB, M, N);

  logResult(TaskType::MATRIX_ADD, Platform::GPU, TPB * TPB, M, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}