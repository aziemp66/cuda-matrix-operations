#include "arg_parser.h"
#include "config.h"
#include "example.h"
#include "logger.h"
#include "task_executor.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
  srand(time(0));

  // Parse command-line arguments
  Config config;
  if (!parseArguments(argc, argv, config)) {
    return 1;
  }

  printf("Using TPB=%d, size=%d\n\n", config.tpb, config.size);

  // Initialize logger
  if (!initLogger(config.cpu_log_path.c_str(), config.gpu_log_path.c_str())) {
    printf("Failed to initialize logger. Exiting.\n");
    return 1;
  }

  // Parse and execute tasks
  std::vector<std::string> tasksToRun;
  if (!config.tasks.empty()) {
    // Parse comma-separated task list
    for (const auto &taskArg : config.tasks) {
      auto parsedTasks = parseTaskList(taskArg.c_str());
      tasksToRun.insert(tasksToRun.end(), parsedTasks.begin(),
                        parsedTasks.end());
    }
  }

  // Execute tasks
  if (tasksToRun.empty()) {
    // Default: run matrixmultnaive CPU and GPU float
    printf("No tasks specified, running default: "
           "matrixmultnaive_cpu_float,matrixmultnaive_gpu_float\n\n");
    float timeCPU = MatrixMultNaiveExampleCPUFloat(config.tpb, config.size);
    float timeGPU = MatrixMultNaiveExampleGPUFloat(config.tpb, config.size);
    printf("\nWinner: %s\n", timeCPU < timeGPU ? "CPU" : "GPU");
    printf("Difference: %f ms\n", timeCPU - timeGPU);
  } else {
    printf("Running %zu task(s):\n", tasksToRun.size());
    for (size_t i = 0; i < tasksToRun.size(); i++) {
      printf("  [%zu] %s\n", i + 1, tasksToRun[i].c_str());
    }
    printf("\n");

    if (!executeTasks(tasksToRun, config.tpb, config.size)) {
      printf("\nWarning: Some tasks failed to execute.\n");
    }
  }

  // Close logger
  closeLogger();

  return 0;
}