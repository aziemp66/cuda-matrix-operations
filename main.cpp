// File: src/main.cpp
#include "example.h"

#include <cstdlib>
#include <ctime>
#include <unistd.h>

using namespace std;

int main() {
  srand(time(0));

  VectorAddExampleFloat();
  sleep(15);

  VectorAddExampleDouble();
  sleep(15);

  MatrixAddExampleFloat();
  sleep(15);

  MatrixAddExampleDouble();
  sleep(15);

  MatrixMultNaiveExampleFloat();
  sleep(15);

  MatrixMultNaiveExampleDouble();

  return 0;
}