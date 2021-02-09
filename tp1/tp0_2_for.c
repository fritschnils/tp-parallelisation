#include <stdio.h>
#include <omp.h>
#define SIZE  100
#define CHUNK  10

int main() {
  int i, tid;
  double a[SIZE], b[SIZE], c[SIZE], sum = 0.;
  
  for (i = 0; i < SIZE; i++)
    a[i] = b[i] = i;

  #pragma omp parallel private(tid) reduction(+: sum)
  {
    tid = omp_get_thread_num();
    if (tid == 0)
      printf("Nb threads = %d\n", omp_get_num_threads());
    printf("Thread %d: starting...\n", tid);

    #pragma omp for
    for (i = 0; i < SIZE; i++) {
      c[i] = a[i] + b[i];
      sum += c[i];
      printf("Thread %d: c[%2d] = %g\n", tid, i, c[i]);
    }
  }
  printf("sum = %g\n", sum);
  return 0;
}
