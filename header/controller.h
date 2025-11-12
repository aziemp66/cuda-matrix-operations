#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <vector>

// Function declarations for vector operations
float VectorAddControllerCPUFloat(std::vector<float> &h_C, int n);
float VectorAddControllerCPUDouble(std::vector<double> &h_C, int n);
float VectorAddControllerGPUFloat(std::vector<float> &h_C, int TPB, int n);
float VectorAddControllerGPUDouble(std::vector<double> &h_C, int TPB, int n);

// Function declarations for matrix addition
float MatrixAddControllerCPUFloat(std::vector<float> &h_C, int n);
float MatrixAddControllerCPUDouble(std::vector<double> &h_C, int n);
float MatrixAddControllerGPUFloat(std::vector<float> &h_C, int TPB, int n);
float MatrixAddControllerGPUDouble(std::vector<double> &h_C, int TPB, int n);

// Function declarations for naive matrix multiplication
float MatrixMultNaiveControllerCPUFloat(std::vector<float> &h_C, int n);
float MatrixMultNaiveControllerCPUDouble(std::vector<double> &h_C, int n);
float MatrixMultNaiveControllerGPUFloat(std::vector<float> &h_C, int TPB,
                                        int n);
float MatrixMultNaiveControllerGPUDouble(std::vector<double> &h_C, int TPB,
                                         int n);

// Function declarations for tiled matrix multiplication
float MatrixMultTiledControllerGPUFloat(std::vector<float> &h_C, int TPB,
                                        int n);
float MatrixMultTiledControllerGPUDouble(std::vector<double> &h_C, int TPB,
                                         int n);

#endif // CONTROLLER_H
