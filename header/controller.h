#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <vector>

// Function declarations for vector operations
float VectorAddControllerCPU(std::vector<float> &h_C, int n);
float VectorAddControllerCPU(std::vector<double> &h_C, int n);
float VectorAddControllerGPU(std::vector<float> &h_C, int TPB, int n);
float VectorAddControllerGPU(std::vector<double> &h_C, int TPB, int n);

// Function declarations for matrix addition
float MatrixAddControllerCPU(std::vector<float> &h_C, int n);
float MatrixAddControllerCPU(std::vector<double> &h_C, int n);
float MatrixAddControllerGPU(std::vector<float> &h_C, int TPB, int n);
float MatrixAddControllerGPU(std::vector<double> &h_C, int TPB, int n);

// Function declarations for naive matrix multiplication
float MatrixMultNaiveControllerCPU(std::vector<float> &h_C, int n);
float MatrixMultNaiveControllerCPU(std::vector<double> &h_C, int n);
float MatrixMultNaiveControllerGPU(std::vector<float> &h_C, int TPB, int n);
float MatrixMultNaiveControllerGPU(std::vector<double> &h_C, int TPB, int n);

// Function declarations for tiled matrix multiplication
float MatrixMultTiledControllerGPU(std::vector<float> &h_C, int TPB, int n);
float MatrixMultTiledControllerGPU(std::vector<double> &h_C, int TPB, int n);

#endif // CONTROLLER_H
