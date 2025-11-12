#ifndef TASK_EXECUTOR_H
#define TASK_EXECUTOR_H

#include <string>
#include <vector>

// Parse a comma-separated task string into individual tasks
// Returns a vector of trimmed task strings
std::vector<std::string> parseTaskList(const char *tasksStr);

// Execute a single task
// Format: {taskname}_{platform}_{type}
// Returns true if successful, false otherwise
bool executeTask(const char *taskStr, int TPB, int n);

bool executeTask(const char *taskStr, int TPB, int n,
                 std::vector<float> h_C_compare);

// Execute multiple tasks
// Returns true if all tasks succeeded, false otherwise
bool executeTasks(const std::vector<std::string> &tasks,
                  const std::vector<int> TPB, int n_exp_start, int n_exp_end,
                  int num_runs);

// Print usage information for task format
void printTaskUsage();

#endif // TASK_EXECUTOR_H
