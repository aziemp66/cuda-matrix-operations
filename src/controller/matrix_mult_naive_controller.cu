#include "controller.h"
#include "cpu.h"
#include "kernels.h"
#include "logger.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

float MatrixMultNaiveControllerCPU(std::vector<float> &h_C, int n) {
  const int M = n;
  const int N = n;
  const int K = n;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  float *h_A = new float[numElementsA];
  float *h_B = new float[numElementsB];
  h_C.resize(numElementsC);

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixMultCPU(h_A, h_B, h_C.data(), M, N, K);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Mult Naive Float Time: %f ms (Size: %d x %d)\n", duration,
         M, N);

  logResult(TaskType::MATRIX_MULT_NAIVE, Platform::CPU, M, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float MatrixMultNaiveControllerGPU(std::vector<float> &h_C, int TPB, int n) {
  const int M = n;
  const int N = n;
  const int K = n;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  float *h_A = new float[numElementsA];
  float *h_B = new float[numElementsB];
  h_C.resize(numElementsC);

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  const size_t sizeA = numElementsA * sizeof(float);
  const size_t sizeB = numElementsB * sizeof(float);
  const size_t sizeC = numElementsC * sizeof(float);

  clock_t start = clock();

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TPB, TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Mult Naive Float Time: %f ms (TPB: %d, Size: %d x %d)\n",
         duration, TPB * TPB, M, N);

  logResult(TaskType::MATRIX_MULT_NAIVE, Platform::GPU, TPB * TPB, M, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}

float MatrixMultNaiveControllerCPU(std::vector<double> &h_C, int n) {
  const int M = n;
  const int N = n;
  const int K = n;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  double *h_A = new double[numElementsA];
  double *h_B = new double[numElementsB];
  h_C.resize(numElementsC);

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  clock_t start = clock();
  matrixMultCPU(h_A, h_B, h_C.data(), M, N, K);
  clock_t end = clock();

  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("CPU Matrix Mult Naive Double Time: %f ms (Size: %d x %d)\n", duration,
         M, N);

  logResult(TaskType::MATRIX_MULT_NAIVE, Platform::CPU, M, duration);

  delete[] h_A;
  delete[] h_B;
  return duration;
}

float MatrixMultNaiveControllerGPU(std::vector<double> &h_C, int TPB, int n) {
  const int M = n;
  const int N = n;
  const int K = n;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  double *h_A = new double[numElementsA];
  double *h_B = new double[numElementsB];
  h_C.resize(numElementsC);

  for (int i = 0; i < numElementsA; ++i) {
    h_A[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < numElementsB; ++i) {
    h_B[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  const size_t sizeA = numElementsA * sizeof(double);
  const size_t sizeB = numElementsB * sizeof(double);
  const size_t sizeC = numElementsC * sizeof(double);

  clock_t start = clock();

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TPB, TPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixMultNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

  clock_t end = clock();
  float duration = 1000.0f * (float)(end - start) / CLOCKS_PER_SEC;
  printf("GPU Matrix Mult Naive Double Time: %f ms (TPB: %d, Size: %d x %d)\n",
         duration, TPB * TPB, M, N);

  logResult(TaskType::MATRIX_MULT_NAIVE, Platform::GPU, TPB * TPB, M, duration);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;

  return duration;
}