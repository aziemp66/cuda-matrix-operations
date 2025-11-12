#include "arg_parser.h"
#include "config.h"
#include "controller.h"
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
    printf("No tasks specified. Exiting.\n");
    printTaskUsage();
    closeLogger();
    return 1;
  } else {
    if (!executeTasks(tasksToRun, config.tpb_list, config.size_expo_range_start,
                      config.size_expo_range_end, config.num_runs)) {
      printf("One or more tasks failed during execution.\n");
      closeLogger();
      return 1;
    }
  }

  // Close logger
  closeLogger();

  return 0;
}