#ifndef CONTROLLER_H
#define CONTROLLER_H

// Function declarations for vector operations
float VectorAddControllerCPUFloat(int TPB, int n);
float VectorAddControllerGPUFloat(int TPB, int n);
float VectorAddControllerCPUDouble(int TPB, int n);
float VectorAddControllerGPUDouble(int TPB, int n);

// Function declarations for matrix addition
float MatrixAddControllerCPUFloat(int TPB, int n);
float MatrixAddControllerGPUFloat(int TPB, int n);
float MatrixAddControllerCPUDouble(int TPB, int n);
float MatrixAddControllerGPUDouble(int TPB, int n);

// Function declarations for naive matrix multiplication
float MatrixMultNaiveControllerCPUFloat(int TPB, int n);
float MatrixMultNaiveControllerGPUFloat(int TPB, int n);
float MatrixMultNaiveControllerCPUDouble(int TPB, int n);
float MatrixMultNaiveControllerGPUDouble(int TPB, int n);

// Function declarations for tiled matrix multiplication
float MatrixMultTiledControllerCPUFloat(int TPB, int n);
float MatrixMultTiledControllerGPUFloat(int TPB, int n);
float MatrixMultTiledControllerCPUDouble(int TPB, int n);
float MatrixMultTiledControllerGPUDouble(int TPB, int n);

#endif // CONTROLLER_H
