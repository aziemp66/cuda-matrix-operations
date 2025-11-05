#include "arg_parser.h"
#include "config.h"

#include <cstdio>
#include <cstring>

bool parseArguments(int argc, char* argv[], Config& config) {
  // Set defaults
  config.tpb = DefaultConfig::DEFAULT_TPB;
  config.size = DefaultConfig::DEFAULT_SIZE;
  config.log_path = DefaultConfig::DEFAULT_LOG_FILE;
  config.tasks.clear();
  
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-tpb=", 5) == 0) {
      config.tpb = atoi(argv[i] + 5);
      if (config.tpb <= 0) {
        printf("Error: TPB must be a positive integer\n");
        return false;
      }
    } else if (strncmp(argv[i], "-size=", 6) == 0) {
      config.size = atoi(argv[i] + 6);
      if (config.size <= 0) {
        printf("Error: size must be a positive integer\n");
        return false;
      }
    } else if (strncmp(argv[i], "-path=", 6) == 0) {
      config.log_path = std::string(argv[i] + 6);
    } else if (strncmp(argv[i], "-task=", 6) == 0) {
      // Task parsing will be done separately using task_executor
      config.tasks.push_back(std::string(argv[i] + 6));
    } else {
      printf("Unknown argument: %s\n", argv[i]);
      printUsage(argv[0]);
      return false;
    }
  }
  
  return true;
}

void printUsage(const char* programName) {
  printf("Usage: %s [-tpb=<value>] [-size=<value>] [-path=<path>] [-task=<tasks>]\n", programName);
  printf("  -tpb=<value>:   Threads per block (int, default: %d)\n", DefaultConfig::DEFAULT_TPB);
  printf("  -size=<value>:  Size parameter (int, default: %d) - for vectors this is the size, for matrices this is n (matrix will be n x n)\n", DefaultConfig::DEFAULT_SIZE);
  printf("  -path=<path>:   Path to log file (string, default: %s)\n", DefaultConfig::DEFAULT_LOG_FILE);
  printf("  -task=<tasks>:  Comma-separated list of tasks to run (e.g., matrixmultnaive_gpu_float,matrixmulttiled_gpu_float)\n");
  printf("                  Format: {taskname}_{platform}_{type}\n");
  printf("                  Task names: vectoradd, matrixadd, matrixmultnaive, matrixmulttiled\n");
  printf("                  Platforms: cpu, gpu\n");
  printf("                  Types: float, double\n");
}

