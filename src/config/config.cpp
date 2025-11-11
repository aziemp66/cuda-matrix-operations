#include "config.h"
#include <vector>

namespace DefaultConfig {
const int DEFAULT_NUM_RUNS = 10;
const int DEFAULT_SIZE_EXPO_RANGE_START = 8;
const int DEFAULT_SIZE_EXPO_RANGE_END = 12;
const char *DEFAULT_CPU_LOG_FILE = "./results/cpu_results.csv";
const char *DEFAULT_GPU_LOG_FILE = "./results/gpu_results.csv";
const std::vector<int> DEFAULT_TPB_LIST = std::vector<int>{8, 32, 64};
} // namespace DefaultConfig