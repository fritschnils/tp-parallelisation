#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define HYPERTHREADING 1      // 1 if hyperthreading is on, 0 otherwise
#define ERROR          1.e-20 // Acceptable precision
#define MAX_VAL        500    // Random values are [0, MAX_VAL]

// Matrix and vector sizes (5120: UHD TV)
#define N 25600

// Reference computation kernel (do not touch)
void bubble_sort_reference(double tab[N]) {
  size_t i, j;
  double temp;

  for (i = 0; i < N ; i++) {
    for (j = 0; j < N - i - 1; j++) {
      if (tab[j] > tab[j + 1]) {
        temp = tab[j + 1];
        tab[j + 1] = tab[j];
        tab[j] = temp;
      }
    }
  }
}
/* Résultats :

Reference time : 1.94877 s
Kernel time -- : 1.99292 s
Speedup ------ : 0.97784
Efficiency --- : 0.48892
0.00290386 0.0444301 0.0904261 0.0909183 0.123984 ... 499.965 499.979 499.98 499.987 499.994 
0.00290386 0.0444301 0.0904261 0.0909183 0.123984 ... 499.965 499.979 499.98 499.987 499.994 
OK results :-)
*/

// Computation kernel (to parallelize)
void bubble_sort_kernel(double tab[N]) {
  size_t i, j;
  double temp;

  #pragma omp for schedule(guided)
  for (i = 0; i < N ; i++) {
    for (j = 0; j < N - i - 1; j++) {
      if (tab[j] > tab[j + 1]) {
        temp = tab[j + 1];
        tab[j + 1] = tab[j];
        tab[j] = temp;
      }
    }
  }
}
/* Résultats :

Reference time : 2.13888 s
Kernel time -- : 1.90986 s
Speedup ------ : 1.11992
Efficiency --- : 0.55996
0.0214903 0.0382192 0.0431424 0.0460404 0.0666443 ... 499.805 499.82 499.887 499.951 499.972 
0.0214903 0.0382192 0.0431424 0.0460404 0.0666443 ... 499.805 499.82 499.887 499.951 499.972 
OK results :-)
*/

void print_sample(double tab[], size_t size, size_t sample_length) {
  if (size <= 2 * sample_length) {
    for (size_t i = 0; i < size; i++)
      printf("%g ", tab[i]);
  } else {
    for (size_t i = 0; (i < size) && (i < sample_length); i++)
      printf("%g ", tab[i]);
    printf("... ");
    for (size_t i = size - sample_length; i < size; i++)
      printf("%g ", tab[i]);
  }
  printf("\n");
}

int main() {
  double* a   = malloc(N * sizeof(double));
  double* ref = malloc(N * sizeof(double));
  double time_reference, time_kernel, speedup, efficiency;

  // Initialization by random values
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < N; i++)
    ref[i] = a[i] = (float)rand()/(float)(RAND_MAX/MAX_VAL);

  time_reference = omp_get_wtime();
  bubble_sort_reference(ref);
  time_reference = omp_get_wtime() - time_reference;
  printf("Reference time : %3.5lf s\n", time_reference);

  time_kernel = omp_get_wtime();
  bubble_sort_kernel(a);
  time_kernel = omp_get_wtime() - time_kernel;
  printf("Kernel time -- : %3.5lf s\n", time_kernel);

  speedup = time_reference / time_kernel;
  efficiency = speedup / (omp_get_num_procs() / (1 + HYPERTHREADING));
  printf("Speedup ------ : %3.5lf\n", speedup);
  printf("Efficiency --- : %3.5lf\n", efficiency);

  print_sample(ref, N, 5);
  print_sample(a, N, 5);

  // Check if the result differs from the reference
  for (size_t i = 0; i < N; i++) {
    if (ref[i] != a[i]) {
      printf("bad results :-(((\n");
      exit(1);
    }
  }
  printf("OK results :-)\n");

  free(a);
  free(ref);
  return 0;
}
