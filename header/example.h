#ifndef EXAMPLE_H
#define EXAMPLE_H

// --- Example Function Declarations ---
// These functions are defined in the .cpp files.

// Vector Addition
float VectorAddExampleCPUFloat();
float VectorAddExampleGPUFloat();
float VectorAddExampleCPUDouble();
float VectorAddExampleGPUDouble();

// Matrix Addition
float MatrixAddExampleCPUFloat();
float MatrixAddExampleGPUFloat();
float MatrixAddExampleCPUDouble();
float MatrixAddExampleGPUDouble();

// Matrix Multiplication (Naive)
float MatrixMultNaiveExampleCPUFloat();
float MatrixMultNaiveExampleGPUFloat();
float MatrixMultNaiveExampleCPUDouble();
float MatrixMultNaiveExampleGPUDouble();

// Matrix Multiplication (Tiled)
float MatrixMultTiledExampleCPUFloat();
float MatrixMultTiledExampleGPUFloat();
float MatrixMultTiledExampleCPUDouble();
float MatrixMultTiledExampleGPUDouble();

#endif // EXAMPLE_H
