#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Default configuration values
namespace DefaultConfig {
extern const int DEFAULT_NUM_RUNS;
extern const int DEFAULT_SIZE_EXPO_RANGE_START;
extern const int DEFAULT_SIZE_EXPO_RANGE_END;
extern const char *DEFAULT_CPU_LOG_FILE;
extern const char *DEFAULT_GPU_LOG_FILE;
extern const std::vector<int> DEFAULT_TPB_LIST;
} // namespace DefaultConfig

struct Config {
  int num_runs = DefaultConfig::DEFAULT_NUM_RUNS;
  int size_expo_range_start = DefaultConfig::DEFAULT_SIZE_EXPO_RANGE_START;
  int size_expo_range_end = DefaultConfig::DEFAULT_SIZE_EXPO_RANGE_END;
  std::string cpu_log_path;
  std::string gpu_log_path;
  std::vector<std::string> tasks;
  std::vector<int> tpb_list;
};

#endif // CONFIG_H
