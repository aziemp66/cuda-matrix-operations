#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include "config.h"

// Parse command-line arguments
// Returns true if parsing was successful, false otherwise
bool parseArguments(int argc, char *argv[], Config &config);

// Print usage information
void printUsage(const char *programName);

#endif // ARG_PARSER_H
