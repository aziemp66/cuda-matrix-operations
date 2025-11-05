#ifndef TASK_EXECUTOR_H
#define TASK_EXECUTOR_H

#include <vector>
#include <string>

// Parse a comma-separated task string into individual tasks
// Returns a vector of trimmed task strings
std::vector<std::string> parseTaskList(const char* tasksStr);

// Execute a single task
// Format: {taskname}_{platform}_{type}
// Returns true if successful, false otherwise
bool executeTask(const char* taskStr, int TPB, int n);

// Execute multiple tasks
// Returns true if all tasks succeeded, false otherwise
bool executeTasks(const std::vector<std::string>& tasks, int TPB, int n);

// Print usage information for task format
void printTaskUsage();

#endif // TASK_EXECUTOR_H

