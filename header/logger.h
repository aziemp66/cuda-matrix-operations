#ifndef LOGGER_H
#define LOGGER_H

#include <cstdio>

// Task types
enum class TaskType {
  VECTOR_ADD,
  MATRIX_ADD,
  MATRIX_MULT_NAIVE,
  MATRIX_MULT_TILED
};

// Platform types
enum class Platform {
  CPU,
  GPU
};

// Initialize logger (opens CSV file, writes header)
void initLogger(const char* filename = "results.csv");

// Log a result entry
void logResult(TaskType taskType, Platform platform, int threadsPerBlock, int size, float duration);

// Close logger (closes file)
void closeLogger();

// Helper function to convert size to bitwise format string
// Returns a string like "1 << 10" for 1024
const char* sizeToBitwiseFormat(int size);

#endif // LOGGER_H

