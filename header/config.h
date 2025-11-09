#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

// Default configuration values
namespace DefaultConfig {
extern const int DEFAULT_TPB;
extern const int DEFAULT_SIZE;
extern const char *DEFAULT_CPU_LOG_FILE;
extern const char *DEFAULT_GPU_LOG_FILE;
} // namespace DefaultConfig

struct Config {
  int tpb = DefaultConfig::DEFAULT_TPB;
  int size = DefaultConfig::DEFAULT_SIZE;
  std::string cpu_log_path;
  std::string gpu_log_path;
  std::vector<std::string> tasks;
};

#endif // CONFIG_H
