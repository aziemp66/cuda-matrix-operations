#include "task_executor.h"
#include "config.h"
#include "example.h"

#include <cctype>
#include <cstdio>
#include <cstring>
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

bool executeTask(const char *taskStr, int TPB, int n) {
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
    printf("  Example: matrixmultnaive_gpu_float\n");
    return false;
  }

  // Normalize strings to lowercase
  toLowercase(taskName);
  toLowercase(platform);
  toLowercase(type);

  // Execute the appropriate function based on task name
  if (strcmp(taskName, "vectoradd") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      if (strcmp(type, "float") == 0) {
        VectorAddExampleCPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        VectorAddExampleCPUDouble(TPB, n);
        return true;
      }
    } else if (strcmp(platform, "gpu") == 0) {
      if (strcmp(type, "float") == 0) {
        VectorAddExampleGPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        VectorAddExampleGPUDouble(TPB, n);
        return true;
      }
    }
  } else if (strcmp(taskName, "matrixadd") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      if (strcmp(type, "float") == 0) {
        MatrixAddExampleCPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        MatrixAddExampleCPUDouble(TPB, n);
        return true;
      }
    } else if (strcmp(platform, "gpu") == 0) {
      if (strcmp(type, "float") == 0) {
        MatrixAddExampleGPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        MatrixAddExampleGPUDouble(TPB, n);
        return true;
      }
    }
  } else if (strcmp(taskName, "matrixmultnaive") == 0) {
    if (strcmp(platform, "cpu") == 0) {
      if (strcmp(type, "float") == 0) {
        MatrixMultNaiveExampleCPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        MatrixMultNaiveExampleCPUDouble(TPB, n);
        return true;
      }
    } else if (strcmp(platform, "gpu") == 0) {
      if (strcmp(type, "float") == 0) {
        MatrixMultNaiveExampleGPUFloat(TPB, n);
        return true;
      } else if (strcmp(type, "double") == 0) {
        MatrixMultNaiveExampleGPUDouble(TPB, n);
        return true;
      }
    }
  } else if (strcmp(taskName, "matrixmulttiled") == 0) {
    if (strcmp(type, "float") == 0) {
      MatrixMultTiledExampleGPUFloat(TPB, n);
      return true;
    } else if (strcmp(type, "double") == 0) {
      MatrixMultTiledExampleGPUDouble(TPB, n);
      return true;
    }
  }

  printf("Error: Unknown task '%s'\n", taskStr);
  printTaskUsage();
  return false;
}

bool executeTasks(const std::vector<std::string> &tasks, int TPB, int n) {
  bool allSuccess = true;

  for (const auto &task : tasks) {
    if (!executeTask(task.c_str(), TPB, n)) {
      allSuccess = false;
    }
  }

  return allSuccess;
}

void printTaskUsage() {
  printf("  Task name must be one of: vectoradd, matrixadd, matrixmultnaive, "
         "matrixmulttiled\n");
  printf("  Platform must be one of: cpu, gpu\n");
  printf("  Type must be one of: float, double\n");
}
