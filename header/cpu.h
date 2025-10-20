#ifndef CPU_H
#define CPU_H

// --- CPU Function Declarations ---

// Vector addition
void vectorAddFloatCPU(const float *A, const float *B, float *C, int N);

void vectorAddDoubleCPU(const double *A, const double *B, double *C, int N);

// float versions
void matrixAddCPU(const float *A, const float *B, float *C, int M, int N);

void matrixMultCPU(const float *A, const float *B, float *C, int M, int N,
                   int K);

// double versions
void matrixAddCPU(const double *A, const double *B, double *C, int M, int N);

void matrixMultCPU(const double *A, const double *B, double *C, int M, int N,
                   int K);

#endif // CPU_H