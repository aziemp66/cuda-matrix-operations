#include "logger.h"

#include <cstdio>
#include <cstring>
#include <cmath>

static FILE* logFile = nullptr;
static bool loggerInitialized = false;

void initLogger(const char* filename) {
  if (loggerInitialized) {
    closeLogger();
  }
  
  logFile = fopen(filename, "w");
  if (logFile == nullptr) {
    printf("Warning: Could not open log file %s for writing\n", filename);
    return;
  }
  
  // Write CSV header
  fprintf(logFile, "task_type,platform,threads_per_block,size,duration_ms\n");
  fflush(logFile);
  loggerInitialized = true;
}

const char* getTaskTypeString(TaskType taskType) {
  switch (taskType) {
    case TaskType::VECTOR_ADD:
      return "vector_add";
    case TaskType::MATRIX_ADD:
      return "matrix_add";
    case TaskType::MATRIX_MULT_NAIVE:
      return "matrixMultNaive";
    case TaskType::MATRIX_MULT_TILED:
      return "MatrixMultTiled";
    default:
      return "unknown";
  }
}

const char* getPlatformString(Platform platform) {
  switch (platform) {
    case Platform::CPU:
      return "CPU";
    case Platform::GPU:
      return "GPU";
    default:
      return "unknown";
  }
}

const char* sizeToBitwiseFormat(int size) {
  // Find the bitwise representation (e.g., 1024 = 1 << 10)
  // We'll find the exponent such that 2^exponent = size
  static char buffer[64];
  
  if (size <= 0) {
    snprintf(buffer, sizeof(buffer), "%d", size);
    return buffer;
  }
  
  // Check if size is a power of 2
  if ((size & (size - 1)) == 0) {
    // It's a power of 2, find the exponent
    int exponent = 0;
    int temp = size;
    while (temp > 1) {
      temp >>= 1;
      exponent++;
    }
    snprintf(buffer, sizeof(buffer), "1 << %d", exponent);
  } else {
    // Not a power of 2, just return the number
    snprintf(buffer, sizeof(buffer), "%d", size);
  }
  
  return buffer;
}

void logResult(TaskType taskType, Platform platform, int threadsPerBlock, int size, float duration) {
  if (!loggerInitialized || logFile == nullptr) {
    printf("Warning: Logger not initialized. Call initLogger() first.\n");
    return;
  }
  
  const char* taskStr = getTaskTypeString(taskType);
  const char* platformStr = getPlatformString(platform);
  const char* sizeStr = sizeToBitwiseFormat(size);
  
  fprintf(logFile, "%s,%s,%d,%s,%f\n", taskStr, platformStr, threadsPerBlock, sizeStr, duration);
  fflush(logFile);
}

void closeLogger() {
  if (logFile != nullptr) {
    fclose(logFile);
    logFile = nullptr;
  }
  loggerInitialized = false;
}

