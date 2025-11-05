#include "example.h"
#include "header/example.h"

#include <cstdlib>
#include <ctime>
#include <unistd.h>

using namespace std;

int main() {
  srand(time(0));

  MatrixMultTiledExampleGPUFloat();

  MatrixMultTiledExampleCPUFloat();

  // MatrixMultTiledExampleGPUDouble();

  // MatrixMultTiledExampleCPUDouble();


  return 0;
}