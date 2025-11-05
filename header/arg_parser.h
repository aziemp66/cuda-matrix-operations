#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <vector>

// Configuration structure to hold parsed arguments
struct Config {
  int tpb = 32;
  int size = 1024;
  std::string log_path;
  std::vector<std::string> tasks;
};

// Parse command-line arguments
// Returns true if parsing was successful, false otherwise
bool parseArguments(int argc, char* argv[], Config& config);

// Print usage information
void printUsage(const char* programName);

#endif // ARG_PARSER_H

