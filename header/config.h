#ifndef CONFIG_H
#define CONFIG_H

// Default configuration values
namespace DefaultConfig {
  constexpr int DEFAULT_TPB = 32;
  constexpr int DEFAULT_SIZE = 1024;  // 1 << 10
  constexpr const char* DEFAULT_LOG_FILE = "results/results.csv";
}

#endif // CONFIG_H

