#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define PI             "3.141592653589793238462"
#define HYPERTHREADING 1      // 1 if hyperthreading is on, 0 otherwise
#define ERROR          1.e-10 // Acceptable precision
#define MAX_VAL        5      // Random values are [0, MAX_VAL]

// Matrix and vector sizes (5120: UHD TV)
#define N 51200000

// Reference computation kernel (do not touch)
void pi_reference(size_t nb_steps, double* pi) {
  double term;
  double sum = 0.;
  double step = 1./(double)nb_steps;

  for (size_t i = 0; i < nb_steps; i++) {
    term = (i + 0.5) * step;
    sum += 4. / (1. + term * term);
  }

  *pi = step * sum;
}
/* Résultats :

Reference time : 0.71500 s
Kernel time -- : 0.71322 s
Speedup ------ : 1.00250
Efficiency --- : 0.50125

Pi (textbook)  : 3.141592653589793238462
Pi reference - : 3.141592653589270422998
Pi parallel -- : 3.141592653589270422998

OK results :-)
*/

// Computation kernel (to parallelize)
void pi_kernel(size_t nb_steps, double* pi) {
  double term;
  double sum = 0.;
  double step = 1./(double)nb_steps;

  #pragma omp parallel for reduction(+:sum) private(term)
  for (size_t i = 0; i < nb_steps; i++) {
    term = (i + 0.5) * step;
    sum += 4. / (1. + term * term);
  }

  *pi = step * sum;
}
/* Résultats :

Reference time : 0.71565 s
Kernel time -- : 0.19072 s
Speedup ------ : 3.75245
Efficiency --- : 1.87622

Pi (textbook)  : 3.141592653589793238462
Pi reference - : 3.141592653589270422998
Pi parallel -- : 3.141592653589777572876

OK results :-)
*/

int main() {
  double pi, pi_ref;
  double time_reference, time_kernel, speedup, efficiency; 
    
  time_reference = omp_get_wtime();
  pi_reference(N, &pi_ref);
  time_reference = omp_get_wtime() - time_reference;
  printf("Reference time : %3.5lf s\n", time_reference);
  
  time_kernel = omp_get_wtime();
  pi_kernel(N, &pi);
  time_kernel = omp_get_wtime() - time_kernel;
  printf("Kernel time -- : %3.5lf s\n", time_kernel);

  speedup = time_reference / time_kernel;
  efficiency = speedup / (omp_get_num_procs() / (1 + HYPERTHREADING));
  printf("Speedup ------ : %3.5lf\n", speedup);
  printf("Efficiency --- : %3.5lf\n", efficiency);

  printf("\nPi (textbook)  : %s\n", PI);
  printf("Pi reference - : %.22g\n", pi_ref);
  printf("Pi parallel -- : %.22g\n\n", pi);

  // Check if the result differs from the reference
  if (fabs(pi_ref - pi) > ERROR) {
    printf("Bad results :-(((\n");
    exit(1);
  }
  printf("OK results :-)\n");
  
  return 0;
}
