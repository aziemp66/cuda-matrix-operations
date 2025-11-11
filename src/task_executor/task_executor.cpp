#include "task_executor.h"
#include "config.h"
#include "controller.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Helper function to normalize string to lowercase
static void toLowercase(char *str) {
  for (char *p = str; *p; p++) {
    *p = static_cast<char>(tolower(static_cast<unsigned char>(*p)));
  }
}

// Helper function to trim whitespace from a string
static std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(" \t\n\r");
  if (first == std::string::npos)
    return "";
  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, last - first + 1);
}

std::vector<std::string> parseTaskList(const char *tasksStr) {
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

float compareResults(float *h_C, float *h_C_compare, int n) {
  float maxError = 0.0f;
  for (int i = 0; i < n; i++) {
    float error = fabs(h_C[i] - h_C_compare[i]);
    if (error > maxError) {
      maxError = error;
    }
  }
  return maxError;
}

double compareResults(double *h_C, double *h_C_compare, int n) {
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

bool executeTask(const char *taskStr, int TPB, int n, int num_runs,
                 float *h_C_compare) {
  // Format: {taskname}_{platform}_{type}
  // e.g., "matrixmultnaive_gpu_float"

  // Make a copy to tokenize
  char taskCopy[256];
  strncpy(taskCopy, taskStr, sizeof(taskCopy) - 1);
  taskCopy[sizeof(taskCopy) - 1] = '\0';

  // Split by underscore
  char *taskName = strtok(taskCopy, "_");
  char *platform = strtok(nullptr, "_");
  char *type = strtok(nullptr, "_");

  if (!taskName || !platform || !type) {
    printf("Error: Invalid task format '%s'. Expected format: "
           "{taskname}_{platform}_{type}\n",
           taskStr);
    printf("  Controller: matrixmultnaive_gpu_float\n");
    return false;
  }

  // Normalize strings to lowercase
  toLowercase(taskName);
  toLowercase(platform);
  toLowercase(type);

  float diff;
  float *h_C;

  // Execute the appropriate function based on task name
  if (strcmp(taskName, "vectoradd") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      VectorAddControllerCPUFloat(h_C, n);
      delete[] h_C;
      return true;
    } else if (strcmp(platform, "gpu") == 0) {
      VectorAddControllerGPUFloat(h_C, TPB, n);
      diff = compareResults(h_C, h_C_compare, n);
      printf("diff error: %f\n", diff);
      delete[] h_C;
      return true;
    }
  } else if (strcmp(taskName, "matrixadd") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      MatrixAddControllerCPUFloat(h_C, n);
      delete[] h_C_compare;
      return true;
    } else if (strcmp(platform, "gpu") == 0) {
      MatrixAddControllerGPUFloat(h_C, TPB, n);
      diff = compareResults(h_C, h_C_compare, n);
      printf("diff error: %f\n", diff);
      delete[] h_C;
      return true;
    }

  } else if (strcmp(taskName, "matrixmultnaive") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      MatrixMultNaiveControllerCPUFloat(h_C, n);

      delete[] h_C;
      return true;
    } else if (strcmp(platform, "gpu") == 0) {
      MatrixMultNaiveControllerGPUFloat(h_C, TPB, n);

      diff = compareResults(h_C, h_C_compare, n);
      printf("diff error: %f\n", diff);
      delete[] h_C;
      return true;
    }
  } else if (strcmp(taskName, "matrixmulttiled") == 0) {
    if (strcmp(platform, "gpu") == 0) {
      MatrixMultTiledControllerGPUFloat(h_C, TPB, n);
      diff = compareResults(h_C, h_C_compare, n);
      printf("diff error: %f\n", diff);
      delete[] h_C;
      return true;
    }
  }

  printf("Error: Unknown task '%s'\n", taskStr);
  printTaskUsage();
  return false;
}

bool executeTasks(const std::vector<std::string> &tasks,
                  const std::vector<int> TPB, int n_exp_start, int n_exp_end,
                  int num_runs) {
  bool allSuccess = true;
  // generate h_C_compare once for all type of tasks
  float *vector_add_h_C_compare;
  float *matrix_add_h_C_compare;
  float *matrix_mult_h_C_compare;

  for (int i = n_exp_start; i <= n_exp_end; i++) {
    printf("=== Executing tasks for size %d ===\n", 1 << i);
    int n = 1 << i;
    // Precompute reference results
    VectorAddControllerCPUFloat(vector_add_h_C_compare, n);
    MatrixAddControllerCPUFloat(matrix_add_h_C_compare, n);
    MatrixMultNaiveControllerCPUFloat(matrix_mult_h_C_compare, n);

    for (const auto &task : tasks) {
      // get platform from task string
      std::string s = task; // make a mutable, owning copy
      std::istringstream iss(s);
      std::string taskName, platform, type;
      std::getline(iss, taskName, '_');
      std::getline(iss, platform, '_');
      std::getline(iss, type, '_');

      if (platform == "cpu") {
        printf("Executing CPU task: %s at size %d\n", task.c_str(), n);
        for (int i = 0; i < num_runs; i++) {
          bool success = executeTask(task.c_str(), 0, n, num_runs, nullptr);
          if (!success) {
            allSuccess = false;
          }
        }
      } else if (platform == "gpu") {
        for (int tpb : TPB) {
          printf("Executing GPU task: %s with TPB=%d at size %d\n",
                 task.c_str(), tpb, n);
          for (int i = 0; i < num_runs; i++) {
            float *h_C_compare;
            if (taskName == "vectoradd") {
              h_C_compare = vector_add_h_C_compare;
            } else if (taskName == "matrixadd") {
              h_C_compare = matrix_add_h_C_compare;
            } else if (taskName == "matrixmultnaive" ||
                       taskName == "matrixmulttiled") {
              h_C_compare = matrix_mult_h_C_compare;
            }

            bool success =
                executeTask(task.c_str(), tpb, n, num_runs, h_C_compare);
            if (!success) {
              allSuccess = false;
            }
          }
        }
      } else {
        printf("Error: Unknown platform in task '%s'\n", task.c_str());
        printTaskUsage();
        allSuccess = false;
      }
    }

    delete[] vector_add_h_C_compare;
    delete[] matrix_add_h_C_compare;
    delete[] matrix_mult_h_C_compare;
  }

  return allSuccess;
}

void printTaskUsage() {
  printf("  Task name must be one of: vectoradd, matrixadd, matrixmultnaive, "
         "matrixmulttiled\n");
  printf("  Platform must be one of: cpu, gpu\n");
  printf("  Type must be one of: float, double\n");
}
