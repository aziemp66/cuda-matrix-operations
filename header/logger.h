#ifndef LOGGER_H
#define LOGGER_H

// Task types
enum class TaskType { VECTOR_ADD, MATRIX_ADD, MATRIX_MULT_NAIVE, MATRIX_MULT_TILED };

// Platform types
enum class Platform { CPU, GPU };

// Data precision types
enum class DataType { FLOAT, DOUBLE };

// Declare headers as external
extern const char* const CPU_EXPECTED_HEADER;
extern const char* const GPU_EXPECTED_HEADER;

// Initialize logger with separate paths for CPU and GPU logs
bool initLogger(const char* cpuFilename = "cpu_results.csv",
                const char* gpuFilename = "gpu_results.csv");

// Log a result entry (automatically chooses correct file based on platform)
// Overload for GPU (includes TPB)
void logResult(TaskType taskType, Platform platform, DataType dtype, int threadsPerBlock, int size,
               float duration);

// Overload for CPU (no TPB)
void logResult(TaskType taskType, Platform platform, DataType dtype, int size, float duration);

// Close both loggers
void closeLogger();

// Helper function to convert size to bitwise format string
const char* sizeToBitwiseFormat(int size);

#endif  // LOGGER_H