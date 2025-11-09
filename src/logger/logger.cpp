#include "logger.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static FILE *cpuLogFile = nullptr;
static FILE *gpuLogFile = nullptr;
static bool loggerInitialized = false;

// Define the headers here
const char *const CPU_EXPECTED_HEADER = "task_type,platform,size,duration_ms";
const char *const GPU_EXPECTED_HEADER =
    "task_type,platform,threads_per_block,size,duration_ms";

// Validate CSV header if file exists (optimized - single file open)
static bool validateHeader(const char *filename, const char *expectedHeader) {
  FILE *testFile = fopen(filename, "r");
  if (testFile == nullptr) {
    // File doesn't exist, which is fine (will create new)
    return true;
  }

  char header[256];
  bool isEmpty = false;
  if (fgets(header, sizeof(header), testFile) == nullptr) {
    // Empty file, treat as new file
    isEmpty = true;
  }

  // Don't close yet - we'll reuse the file handle if valid
  if (!isEmpty) {
    // Remove newline if present
    size_t len = strlen(header);
    if (len > 0 && header[len - 1] == '\n') {
      header[len - 1] = '\0';
    }

    // Compare with expected header
    if (strcmp(header, expectedHeader) != 0) {
      fclose(testFile);
      printf("Error: CSV file '%s' has invalid header format.\n", filename);
      printf("  Expected: %s\n", expectedHeader);
      printf("  Found:    %s\n", header);
      printf("  Please fix the file or use a different path.\n");
      return false;
    }
  }

  fclose(testFile);
  return true;
}

bool initLogger(const char *cpuFilename, const char *gpuFilename) {
  if (loggerInitialized) {
    closeLogger();
  }

  // Validate both files
  if (!validateHeader(cpuFilename, CPU_EXPECTED_HEADER) ||
      !validateHeader(gpuFilename, GPU_EXPECTED_HEADER)) {
    return false;
  }

  // Open CPU log file
  cpuLogFile = fopen(cpuFilename, "a");
  if (cpuLogFile == nullptr) {
    printf("Error: Could not open CPU log file %s for writing\n", cpuFilename);
    return false;
  }

  // Open GPU log file
  gpuLogFile = fopen(gpuFilename, "a");
  if (gpuLogFile == nullptr) {
    printf("Error: Could not open GPU log file %s for writing\n", gpuFilename);
    fclose(cpuLogFile);
    cpuLogFile = nullptr;
    return false;
  }

  // Write headers if files are new
  fseek(cpuLogFile, 0, SEEK_END);
  if (ftell(cpuLogFile) == 0) {
    fprintf(cpuLogFile, "%s\n", CPU_EXPECTED_HEADER);
    fflush(cpuLogFile);
  }

  fseek(gpuLogFile, 0, SEEK_END);
  if (ftell(gpuLogFile) == 0) {
    fprintf(gpuLogFile, "%s\n", GPU_EXPECTED_HEADER);
    fflush(gpuLogFile);
  }

  loggerInitialized = true;
  return true;
}

const char *getTaskTypeString(TaskType taskType) {
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

const char *getPlatformString(Platform platform) {
  switch (platform) {
  case Platform::CPU:
    return "CPU";
  case Platform::GPU:
    return "GPU";
  default:
    return "unknown";
  }
}

const char *sizeToBitwiseFormat(int size) {
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

void logResult(TaskType taskType, Platform platform, int threadsPerBlock,
               int size, float duration) {
  if (!loggerInitialized || (cpuLogFile == nullptr && gpuLogFile == nullptr)) {
    printf("Warning: Logger not initialized. Call initLogger() first.\n");
    return;
  }

  const char *taskStr = getTaskTypeString(taskType);
  const char *platformStr = getPlatformString(platform);
  const char *sizeStr = sizeToBitwiseFormat(size);

  if (platform == Platform::CPU) {
    fprintf(cpuLogFile, "%s,%s,%s,%f\n", taskStr, platformStr, sizeStr,
            duration);
    fflush(cpuLogFile);
  } else {
    fprintf(gpuLogFile, "%s,%s,%d,%s,%f\n", taskStr, platformStr,
            threadsPerBlock, sizeStr, duration);
    fflush(gpuLogFile);
  }
}

void closeLogger() {
  if (cpuLogFile != nullptr) {
    fclose(cpuLogFile);
    cpuLogFile = nullptr;
  }
  if (gpuLogFile != nullptr) {
    fclose(gpuLogFile);
    gpuLogFile = nullptr;
  }
  loggerInitialized = false;
}
