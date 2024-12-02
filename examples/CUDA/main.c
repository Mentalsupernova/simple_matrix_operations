#define NVIDIA_GPU_DEVICE
#define SIMPLE_MATRIX_OPERATIONS_IMPLEMETATION
#include <stdio.h>
#include <math.h>
#include "../../simple_matrix_operations.h"


#define N 2

int main(void) {
  size_t n_dims = 2;
  size_t dims[2] = {N,N};
  simple_matrix  * A = allocate_matrix(n_dims, dims, RANDOM, 1.0, 10);
  simple_matrix  * C = allocate_matrix(n_dims, dims, ZEROS, 0.0, 0.0);
  dot(A,A,C);
  printf("ORIGINAL A MATRIX\n");
  print_matrix(A);

  printf("RESULT OF DOT\n");
  print_matrix(C);
  return 0;
}
