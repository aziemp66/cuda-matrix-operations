#include "example.h"
#include "header/example.h"

#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <cstdio>
#include <cstring>

using namespace std;

int main(int argc, char *argv[]) {
  srand(time(0));

  // Default values
  int TPB = 32;
  int n = 1024;  // 1 << 10

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-tpb=", 5) == 0) {
      TPB = atoi(argv[i] + 5);
      if (TPB <= 0) {
        printf("Error: TPB must be a positive integer\n");
        return 1;
      }
    } else if (strncmp(argv[i], "-size=", 6) == 0) {
      n = atoi(argv[i] + 6);
      if (n <= 0) {
        printf("Error: size must be a positive integer\n");
        return 1;
      }
    } else {
      printf("Unknown argument: %s\n", argv[i]);
      printf("Usage: %s [-tpb=<value>] [-size=<value>]\n", argv[0]);
      printf("  -tpb=<value>:   Threads per block (int, default: 32)\n");
      printf("  -size=<value>:  Size parameter (int, default: 1024) - for vectors this is the size, for matrices this is n (matrix will be n x n)\n");
      return 1;
    }
  }

  printf("Using TPB=%d, size=%d\n\n", TPB, n);

  float timeCPU = MatrixMultNaiveExampleCPUFloat(TPB, n);

  float timeGPU = MatrixMultNaiveExampleGPUFloat(TPB, n);

  printf("Winner: %s\n", timeCPU < timeGPU ? "CPU" : "GPU");
  printf("\ndifference: %f\n", timeCPU - timeGPU);

  // timeCPU = MatrixMultTiledExampleCPUFloat(TPB, n);

  // timeGPU = MatrixMultTiledExampleGPUFloat(TPB, n);

  // printf("Winner: %s\n", timeCPU < timeGPU ? "CPU" : "GPU");
  // printf("\ndifference: %f\n", timeCPU - timeGPU);

  return 0;
}