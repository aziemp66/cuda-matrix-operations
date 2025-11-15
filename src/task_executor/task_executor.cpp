#include "task_executor.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "controller.h"
#include "cpu.h"

// Helper function to normalize string to lowercase
static void toLowercase(char* str) {
  for (char* p = str; *p; p++) {
    *p = static_cast<char>(tolower(static_cast<unsigned char>(*p)));
  }
}

// Helper function to trim whitespace from a string
static std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, last - first + 1);
}

std::vector<std::string> parseTaskList(const char* tasksStr) {
  std::vector<std::string> tasks;
  std::string taskStr = tasksStr;
  size_t start = 0;
  size_t pos = 0;

  while ((pos = taskStr.find(',', start)) != std::string::npos) {
    std::string task = taskStr.substr(start, pos - start);
    task = trim(task);
    if (!task.empty()) {
      tasks.push_back(task);
    }
    start = pos + 1;
  }

  // Add the last task
  std::string task = taskStr.substr(start);
  task = trim(task);
  if (!task.empty()) {
    tasks.push_back(task);
  }

  return tasks;
}

// --- CORRECTED ---
// Added 'const' to both pointer arguments
float compareResults(const float* h_C, const float* h_C_compare, int n) {
  float maxError = 0.0f;
  for (int i = 0; i < n; i++) {
    float error = std::fabs(h_C[i] - h_C_compare[i]);
    if (error > maxError) {
      maxError = error;
    }
  }
  return maxError;
}

// --- CORRECTED ---
// Added 'const' to both pointer arguments
double compareResults(const double* h_C, const double* h_C_compare, int n) {
  // make room for rounding errors
  double maxError = 0.0;
  for (int i = 0; i < n; i++) {
    double error = fabs(h_C[i] - h_C_compare[i]);
    if (error > maxError) {
      maxError = error;
    }
  }
  return maxError;
}

// --- FIX 2: Changed signature to accept const references (solves compile
// error) ---
bool executeTask(const char* taskStr, int TPB, int n, int num_runs,
                 const std::vector<float>& h_C_compare_float,
                 const std::vector<double>& h_C_compare_double) {
  // Format: {taskname}_{platform}_{type}
  // e.g., "matrixmultnaive_gpu_float"

  // Make a copy to tokenize
  char taskCopy[256];
  strncpy(taskCopy, taskStr, sizeof(taskCopy) - 1);
  taskCopy[sizeof(taskCopy) - 1] = '\0';

  // Split by underscore
  char* taskName = strtok(taskCopy, "_");
  char* platform = strtok(nullptr, "_");
  char* type = strtok(nullptr, "_");

  if (!taskName || !platform || !type) {
    printf(
        "Error: Invalid task format '%s'. Expected format: "
        "{taskname}_{platform}_{type}\n",
        taskStr);
    printf("  Controller: matrixmultnaive_gpu_float\n");
    return false;
  }

  // Normalize strings to lowercase
  toLowercase(taskName);
  toLowercase(platform);
  toLowercase(type);

  // We handle floats and doubles in completely separate blocks
  if (strcmp(type, "float") == 0) {
    std::vector<float> h_C;
    float diff;

    if (strcmp(taskName, "vectoradd") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        VectorAddControllerCPUFloat(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        VectorAddControllerGPUFloat(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixadd") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        MatrixAddControllerCPUFloat(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        MatrixAddControllerGPUFloat(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixmultnaive") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        MatrixMultNaiveControllerCPUFloat(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        MatrixMultNaiveControllerGPUFloat(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixmulttiled") == 0) {
      if (strcmp(platform, "gpu") == 0) {
        MatrixMultTiledControllerGPUFloat(h_C, TPB, n, true);
      } else {
        printf("Error: matrixmulttiled is only implemented for GPU platform\n");
        return false;
      }
    } else {
      printf("Error: Unknown task '%s'\n", taskName);
      return false;
    }

  } else if (strcmp(type, "double") == 0) {
    std::vector<double> h_C;
    double diff;

    if (strcmp(taskName, "vectoradd") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        VectorAddControllerCPUDouble(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        VectorAddControllerGPUDouble(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixadd") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        MatrixAddControllerCPUDouble(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        MatrixAddControllerGPUDouble(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixmultnaive") == 0) {
      if (strcmp(platform, "cpu") == 0) {
        MatrixMultNaiveControllerCPUDouble(h_C, n, true);
      } else if (strcmp(platform, "gpu") == 0) {
        MatrixMultNaiveControllerGPUDouble(h_C, TPB, n, true);
      }
    } else if (strcmp(taskName, "matrixmulttiled") == 0) {
      if (strcmp(platform, "gpu") == 0) {
        MatrixMultTiledControllerGPUDouble(h_C, TPB, n, true);
      } else {
        printf("Error: matrixmulttiled is only implemented for GPU platform\n");
        return false;
      }
    } else {
      printf("Error: Unknown task '%s'\n", taskName);
      return false;
    }

  } else {
    printf("Error: Unknown type '%s'\n", type);
    return false;
  }

  return true;
}

bool executeTasks(const std::vector<std::string>& tasks, const std::vector<int> TPB,
                  int n_exp_start, int n_exp_end, int num_runs) {
  bool allSuccess = true;

  // Create reference vectors for BOTH types
  std::vector<float> vector_add_h_C_compare_float;
  std::vector<float> matrix_add_h_C_compare_float;
  std::vector<float> matrix_mult_h_C_compare_float;

  std::vector<double> vector_add_h_C_compare_double;
  std::vector<double> matrix_add_h_C_compare_double;
  std::vector<double> matrix_mult_h_C_compare_double;

  // --- FIX 3: Declared as persistent, empty objects, not references ---
  static const std::vector<float> empty_float;
  static const std::vector<double> empty_double;

  for (int i = n_exp_start; i <= n_exp_end; i++) {
    int n = 1 << i;
    printf("=== Executing tasks for size %d ===\n", n);

    // Generate reference results
    // int max_n = 1 << n;
    // VectorAddControllerCPUFloat(vector_add_h_C_compare_float, max_n, false);
    // MatrixAddControllerCPUFloat(matrix_add_h_C_compare_float, max_n, false);
    // MatrixMultNaiveControllerCPUFloat(matrix_mult_h_C_compare_float, max_n, false);

    // VectorAddControllerCPUDouble(vector_add_h_C_compare_double, max_n, false);
    // MatrixAddControllerCPUDouble(matrix_add_h_C_compare_double, max_n, false);
    // MatrixMultNaiveControllerCPUDouble(matrix_mult_h_C_compare_double, max_n, false);

    for (const auto& task : tasks) {
      std::string s = task;
      std::istringstream iss(s);
      std::string taskName, platform, type;
      std::getline(iss, taskName, '_');
      std::getline(iss, platform, '_');
      std::getline(iss, type, '_');

      if (platform == "cpu") {
        printf("Executing CPU task: %s at size %d\n", task.c_str(), n);
        for (int j = 0; j < num_runs; j++) {
          // Pass the empty reference vectors
          bool success = executeTask(task.c_str(), 0, n, num_runs, empty_float, empty_double);
          if (!success) allSuccess = false;
        }
      } else if (platform == "gpu") {
        for (int tpb : TPB) {
          printf("Executing GPU task: %s with TPB=%d at size %d\n", task.c_str(), tpb, n);
          for (int j = 0; j < num_runs; j++) {
            // Select the correct compare vector based on type
            std::vector<float> h_C_compare_float_task;
            std::vector<double> h_C_compare_double_task;

            if (type == "float") {
              if (taskName == "vectoradd")
                h_C_compare_float_task = vector_add_h_C_compare_float;
              else if (taskName == "matrixadd")
                h_C_compare_float_task = matrix_add_h_C_compare_float;
              else if (taskName == "matrixmultnaive" || taskName == "matrixmulttiled")
                h_C_compare_float_task = matrix_mult_h_C_compare_float;
            } else if (type == "double") {
              if (taskName == "vectoradd")
                h_C_compare_double_task = vector_add_h_C_compare_double;
              else if (taskName == "matrixadd")
                h_C_compare_double_task = matrix_add_h_C_compare_double;
              else if (taskName == "matrixmultnaive" || taskName == "matrixmulttiled")
                h_C_compare_double_task = matrix_mult_h_C_compare_double;
            }

            bool success = executeTask(task.c_str(), tpb, n, num_runs, h_C_compare_float_task,
                                       h_C_compare_double_task);
            if (!success) allSuccess = false;
          }
        }
      } else {
        printf("Error: Unknown platform in task '%s'\n", task.c_str());
        printTaskUsage();
        allSuccess = false;
      }
    }
  }

  return allSuccess;
}

void printTaskUsage() {
  printf(
      "  Task name must be one of: vectoradd, matrixadd, matrixmultnaive, "
      "matrixmulttiled\n");
  printf("  Platform must be one of: cpu, gpu\n");
  printf("  Type must be one of: float, double\n");
}