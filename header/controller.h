#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <vector>

// Function declarations for vector operations
float VectorAddControllerCPUFloat(std::vector<float>& h_C, int n, bool isLogged);
float VectorAddControllerCPUDouble(std::vector<double>& h_C, int n, bool isLogged);
float VectorAddControllerGPUFloat(std::vector<float>& h_C, int TPB, int n, bool isLogged);
float VectorAddControllerGPUDouble(std::vector<double>& h_C, int TPB, int n, bool isLogged);

// Function declarations for matrix addition
float MatrixAddControllerCPUFloat(std::vector<float>& h_C, int n, bool isLogged);
float MatrixAddControllerCPUDouble(std::vector<double>& h_C, int n, bool isLogged);
float MatrixAddControllerGPUFloat(std::vector<float>& h_C, int TPB, int n, bool isLogged);
float MatrixAddControllerGPUDouble(std::vector<double>& h_C, int TPB, int n, bool isLogged);

// Function declarations for naive matrix multiplication
float MatrixMultNaiveControllerCPUFloat(std::vector<float>& h_C, int n, bool isLogged);
float MatrixMultNaiveControllerCPUDouble(std::vector<double>& h_C, int n, bool isLogged);
float MatrixMultNaiveControllerGPUFloat(std::vector<float>& h_C, int TPB, int n, bool isLogged);
float MatrixMultNaiveControllerGPUDouble(std::vector<double>& h_C, int TPB, int n, bool isLogged);

// Function declarations for tiled matrix multiplication
float MatrixMultTiledControllerGPUFloat(std::vector<float>& h_C, int TPB, int n, bool isLogged);
float MatrixMultTiledControllerGPUDouble(std::vector<double>& h_C, int TPB, int n, bool isLogged);

#endif  // CONTROLLER_H
