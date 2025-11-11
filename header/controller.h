#ifndef CONTROLLER_H
#define CONTROLLER_H

// Function declarations for vector operations
float VectorAddControllerCPUFloat(float *h_C, int n);
float VectorAddControllerCPUFloat(double *h_C, int n);
float VectorAddControllerGPUFloat(float *h_C, int TPB, int n);
float VectorAddControllerGPUDouble(double *h_C, int TPB, int n);

// Function declarations for matrix addition
float MatrixAddControllerCPUFloat(float *h_C, int n);
float MatrixAddControllerCPUFloat(double *h_C, int n);
float MatrixAddControllerGPUFloat(float *h_C, int TPB, int n);
float MatrixAddControllerGPUDouble(double *h_C, int TPB, int n);

// Function declarations for naive matrix multiplication
float MatrixMultNaiveControllerCPUFloat(float *h_C, int n);
float MatrixMultNaiveControllerCPUFloat(double *h_C, int n);
float MatrixMultNaiveControllerGPUFloat(float *h_C, int TPB, int n);
float MatrixMultNaiveControllerGPUDouble(double *h_C, int TPB, int n);

// Function declarations for tiled matrix multiplication
float MatrixMultTiledControllerGPUFloat(float *h_C, int TPB, int n);
float MatrixMultTiledControllerGPUDouble(double *h_C, int TPB, int n);

#endif // CONTROLLER_H
