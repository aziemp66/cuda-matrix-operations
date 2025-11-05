#include "logger.h"

#include <cstdio>
#include <cstring>
#include <cmath>

static FILE* logFile = nullptr;
static bool loggerInitialized = false;

const char* EXPECTED_HEADER = "task_type,platform,threads_per_block,size,duration_ms";

// Check if file exists
static bool fileExists(const char* filename) {
  FILE* testFile = fopen(filename, "r");
  if (testFile != nullptr) {
    fclose(testFile);
    return true;
  }
  return false;
}

// Validate CSV header if file exists
static bool validateHeader(const char* filename) {
  FILE* testFile = fopen(filename, "r");
  if (testFile == nullptr) {
    // File doesn't exist, which is fine (will create new)
    return true;
  }
  
  char header[256];
  if (fgets(header, sizeof(header), testFile) == nullptr) {
    // Empty file, treat as new file
    fclose(testFile);
    return true;
  }
  
  fclose(testFile);
  
  // Remove newline if present
  size_t len = strlen(header);
  if (len > 0 && header[len - 1] == '\n') {
    header[len - 1] = '\0';
  }
  
  // Compare with expected header
  if (strcmp(header, EXPECTED_HEADER) != 0) {
    printf("Error: CSV file '%s' has invalid header format.\n", filename);
    printf("  Expected: %s\n", EXPECTED_HEADER);
    printf("  Found:    %s\n", header);
    printf("  Please fix the file or use a different path.\n");
    return false;
  }
  
  return true;
}

bool initLogger(const char* filename) {
  if (loggerInitialized) {
    closeLogger();
  }
  
  // Validate existing file if it exists
  if (fileExists(filename)) {
    if (!validateHeader(filename)) {
      return false;
    }
  }
  
  // Open in append mode (creates file if it doesn't exist)
  logFile = fopen(filename, "a");
  if (logFile == nullptr) {
    printf("Error: Could not open log file %s for writing\n", filename);
    return false;
  }
  
  // Check if file is new (empty or just created)
  // In append mode, file position is at the end, so we need to check file size
  fseek(logFile, 0, SEEK_END);
  long fileSize = ftell(logFile);
  if (fileSize == 0) {
    // New file, write header
    fprintf(logFile, "%s\n", EXPECTED_HEADER);
    fflush(logFile);
  }
  
  loggerInitialized = true;
  return true;
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

