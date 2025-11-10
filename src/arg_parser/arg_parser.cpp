#include "arg_parser.h"
#include "config.h"

#include <cstdio>
#include <cstring>

bool parseArguments(int argc, char *argv[], Config &config) {
  // Set defaults
  config.tpb = DefaultConfig::DEFAULT_TPB;
  config.size = DefaultConfig::DEFAULT_SIZE;
  config.cpu_log_path = DefaultConfig::DEFAULT_CPU_LOG_FILE;
  config.gpu_log_path = DefaultConfig::DEFAULT_GPU_LOG_FILE;
  config.tasks.clear();

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-tpb=", 5) == 0) {
      config.tpb = atoi(argv[i] + 5);
      if (config.tpb <= 0) {
        printf("Error: TPB must be a positive integer\n");
        return false;
      }
    } else if (strncmp(argv[i], "-size=", 6) == 0) {
      const char *sizeStr = argv[i] + 6;
      int base, exponent;

      // Try parsing bitwise shift (e.g., "1 << 10")
      if (sscanf(sizeStr, "%d << %d", &base, &exponent) == 2) {
        config.size = base << exponent;
      }
      // Try parsing bitwise shift with no spaces (e.g., "1<<10")
      else if (sscanf(sizeStr, "%d<<%d", &base, &exponent) == 2) {
        config.size = base << exponent;
      }
      // Try parsing as a plain integer
      else {
        config.size = atoi(sizeStr);
        // Basic check for atoi failure (e.g., "abc")
        if (config.size == 0 && sizeStr[0] != '0') {
          printf("Error: Invalid size format: %s\n", sizeStr);
          printf("  Must be an integer (e.g., 1024) or bitwise shift (e.g., "
                 "\"1 << 10\").\n");
          return false;
        }
      }

      if (config.size <= 0) {
        printf("Error: size must be a positive integer\n");
        return false;
      }
    } else if (strncmp(argv[i], "-cpu_path=", 10) == 0) {
      config.cpu_log_path = std::string(argv[i] + 10);
    } else if (strncmp(argv[i], "-gpu_path=", 10) == 0) {
      config.gpu_log_path = std::string(argv[i] + 10);
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

void printUsage(const char *programName) {
  printf("Usage: %s [-tpb=<value>] [-size=<value>] [-cpu_path=<path>] "
         "[-gpu_path=<path>] [-task=<tasks>]\n",
         programName);
  printf("  -tpb=<value>:      Threads per block (int, default: %d)\n",
         DefaultConfig::DEFAULT_TPB);
  printf("  -size=<value>:     Size parameter (int, default: %d). Accepts "
         "integers (e.g., 1024)\n"
         "                     or bitwise shifts (e.g., \"1 << 10\").\n",
         DefaultConfig::DEFAULT_SIZE);
  printf("                     For vectors this is the size, for matrices this "
         "is n (matrix will be n x n)\n");
  printf("  -cpu_path=<path>:  Path to CPU log file (string, default: %s)\n",
         DefaultConfig::DEFAULT_CPU_LOG_FILE);
  printf("  -gpu_path=<path>:  Path to GPU log file (string, default: %s)\n",
         DefaultConfig::DEFAULT_GPU_LOG_FILE);
  printf("  -task=<tasks>:     Comma-separated list of tasks to run\n");
  printf("                     Format: {taskname}_{platform}_{type}\n");
  printf("                     Task names: vectoradd, matrixadd, "
         "matrixmultnaive, matrixmulttiled\n");
  printf("                     Platforms: cpu, gpu\n");
  printf("                     Types: float, double\n");
  printf("\nController:\n");
  printf("  %s -tpb=16 -size=\"1 << 10\" -cpu_path=cpu_results.csv "
         "-gpu_path=gpu_results.csv\n",
         programName);
  printf("  %s -task=matrixmulttiled_gpu_float,matrixmulttiled_cpu_float\n",
         programName);
}