#ifndef EXAMPLE_H
#define EXAMPLE_H

// --- Example Function Declarations ---
// These functions are defined in the .cpp files.

// Vector Addition
float VectorAddExampleCPUFloat(int TPB, int n);
float VectorAddExampleGPUFloat(int TPB, int n);
float VectorAddExampleCPUDouble(int TPB, int n);
float VectorAddExampleGPUDouble(int TPB, int n);

// Matrix Addition
float MatrixAddExampleCPUFloat(int TPB, int n);
float MatrixAddExampleGPUFloat(int TPB, int n);
float MatrixAddExampleCPUDouble(int TPB, int n);
float MatrixAddExampleGPUDouble(int TPB, int n);

// Matrix Multiplication (Naive)
float MatrixMultNaiveExampleCPUFloat(int TPB, int n);
float MatrixMultNaiveExampleGPUFloat(int TPB, int n);
float MatrixMultNaiveExampleCPUDouble(int TPB, int n);
float MatrixMultNaiveExampleGPUDouble(int TPB, int n);

// Matrix Multiplication (Tiled)
float MatrixMultTiledExampleGPUFloat(int TPB, int n);
float MatrixMultTiledExampleGPUDouble(int TPB, int n);

#endif // EXAMPLE_H
