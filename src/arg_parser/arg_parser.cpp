#include "arg_parser.h"
#include "config.h"

#include <cstdio>
#include <cstring>
#include <sstream> // Added for thread-safe string splitting
#include <string>
#include <vector>

bool parseArguments(int argc, char *argv[], Config &config) {
  // Set defaults
  config.cpu_log_path = DefaultConfig::DEFAULT_CPU_LOG_FILE;
  config.gpu_log_path = DefaultConfig::DEFAULT_GPU_LOG_FILE;
  config.size_expo_range_start = DefaultConfig::DEFAULT_SIZE_EXPO_RANGE_START;
  config.size_expo_range_end = DefaultConfig::DEFAULT_SIZE_EXPO_RANGE_END;
  config.num_runs = DefaultConfig::DEFAULT_NUM_RUNS;
  config.tpb_list = DefaultConfig::DEFAULT_TPB_LIST;
  config.tasks.clear();

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-tpb=", 5) == 0) {
      // FIXED: Replaced non thread-safe strtok with std::stringstream
      config.tpb_list.clear();
      std::string tpbStr = argv[i] + 5;
      std::stringstream ss(tpbStr);
      std::string token;

      while (std::getline(ss, token, ',')) {
        try {
          int tpbValue = std::stoi(token);
          if (tpbValue < 1) {
            printf("Error: TPB values must be greater than 0\n");
            return false;
          }
          config.tpb_list.push_back(tpbValue);
        } catch (...) {
          printf("Error: Invalid non-integer value in TPB list: %s\n",
                 token.c_str());
          return false;
        }
      }

    } else if (strncmp(argv[i], "-size_exp_range=", 16) == 0) {
      const char *sizeStr = argv[i] + 16;
      int startRange, endRange;
      if (sscanf(sizeStr, "%d:%d", &startRange, &endRange) == 2) {
        if (startRange < 0 || endRange < 0 || startRange > endRange) {
          printf("Error: Invalid size_expo_range values. Ensure 0 <= start <= "
                 "end.\n");
          return false;
        }
        config.size_expo_range_start = startRange;
        config.size_expo_range_end = endRange;
      } else {
        printf("Error: Invalid size_expo_range format. Use start:end (e.g., "
               "8:12)\n");
        return false;
      }

    } else if (strncmp(argv[i], "-cpu_path=", 10) == 0) {
      config.cpu_log_path = std::string(argv[i] + 10);

    } else if (strncmp(argv[i], "-gpu_path=", 10) == 0) {
      config.gpu_log_path = std::string(argv[i] + 10);

    } else if (strncmp(argv[i], "-task=", 6) == 0) {
      config.tasks.push_back(std::string(argv[i] + 6));

    } else if (strncmp(argv[i], "-num_runs=", 10) == 0) {
      int runs = atoi(argv[i] + 10);
      if (runs <= 0) {
        printf("Error: num_runs must be a positive integer\n");
        return false;
      }
      config.num_runs = runs;
    } else {
      printf("Unknown argument: %s\n", argv[i]);
      // Ensure printUsage is declared in arg_parser.h
      printUsage(argv[0]);
      return false;
    }
  }

  return true;
}

void printUsage(const char *programName) {
  printf("Usage: %s [options]\n\n", programName);
  printf("Options:\n");
  printf("  -tpb=<list>            Comma-separated list of threads per block "
         "to test.\n");
  printf("                         Example: -tpb=16,32,64,128\n");
  printf("\n");
  printf("  -size_exp_range=<s:e>  Range of exponents for matrix sizes (2^s to "
         "2^e).\n");
  printf("                         Example: -size_exp_range=10:20 tests sizes "
         "2^10 to 2^20\n");
  printf("\n");
  printf("  -num_runs=<n>          Number of times to repeat each test for "
         "averaging.\n");
  printf("                         Example: -num_runs=5\n");
  printf("  -task=<name>           Specific kernel/task to run. Can be used "
         "multiple times.\n");
  printf("                         Example: -task=naive -task=tiled\n");
  printf("\n");
  printf("  -cpu_path=<path>       File path for saving standard CPU logs.\n");
  printf("  -gpu_path=<path>       File path for saving standard GPU logs.\n");
  printf("\n");
  printf("  -task=<tasks>:     Comma-separated list of tasks to run\n");
  printf("                     Format: {taskname}_{platform}_{type}\n");
  printf("                     Task names: vectoradd, matrixadd, "
         "matrixmultnaive, matrixmulttiled\n");
  printf("                     Platforms: cpu, gpu\n");
  printf("                     Types: float, double\n\n");
}