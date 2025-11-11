#ifndef LOGGER_H
#define LOGGER_H

enum class TaskType {
  VECTOR_ADD,
  MATRIX_ADD,
  MATRIX_MULT_NAIVE,
  MATRIX_MULT_TILED
};

enum class Platform { CPU, GPU };

// Declare headers as external
extern const char *const CPU_EXPECTED_HEADER;
extern const char *const GPU_EXPECTED_HEADER;

// Initialize logger with separate paths for CPU and GPU logs
bool initLogger(const char *cpuFilename = "cpu_results.csv",
                const char *gpuFilename = "gpu_results.csv");

// Log a result entry (automatically chooses correct file based on platform)
void logResult(TaskType taskType, Platform platform, int threadsPerBlock,
               int size, float duration);

void logResult(TaskType taskType, Platform platform, int size, float duration);

// Close both loggers
void closeLogger();

const char *sizeToBitwiseFormat(int size);

#endif // LOGGER_H
