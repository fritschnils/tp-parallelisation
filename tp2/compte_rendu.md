# Compte rendu TP2 Programmation Parallèle

## 1) Produit matrice-matrice

**Sans parallélisation**
```c
void matmat_reference(double C[N][N], double A[N][N], double B[N][N]) {
  size_t i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i][j] = 0.;
      for (k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
```
On parallélise la boucle for (en testant avec 8 threads)

**Avec parallélisation**
```c
void matmat_kernel(double C[N][N], double A[N][N], double B[N][N]) {
  size_t i, j, k;
  #pragma omp parallel num_threads(128)
  {
    #pragma omp for schedule(dynamic,N/128) private(j,k) collapse(2)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        C[i][j] = 0.;
        for (k = 0; k < N; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
}
```

### 1.1 Calcul de Pi

**Sans parallélisation**
```c
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
```
On parallélise la boucle for avec une **reduction sur la variable *sum***. Il faut veiller à mettre la variable ***term* en private**

**Avec parallélisation**
```c
void pi_kernel(size_t nb_steps, double* pi) {
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

Reference time : 0.71565 s
Kernel time -- : 0.19072 s
Speedup ------ : 3.75245
Efficiency --- : 1.87622

Pi (textbook)  : 3.141592653589793238462
Pi reference - : 3.141592653589270422998
Pi parallel -- : 3.141592653589777572876

OK results :-)
*/
```

### 1.2 Tri par énumération

**Sans parallélisation**
```c
void enumeration_sort_reference(double tab[N]) {
  size_t i, j;
  size_t* position = malloc(N * sizeof(size_t));
  double* copy     = malloc(N * sizeof(size_t));

  for (i = 0; i < N; i++) {
    position[i] = 0;
    copy[i] = tab[i];
  }
  
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
      if ((tab[j] < tab[i]) || ((tab[i] == tab[j]) && (i < j))) {
        position[i]++;
      }
    }
  }

  for (i = 0; i < N; i++)
    tab[position[i]] = copy[i];

  free(position);
  free(copy);

}
/* Résultats :

Reference time : 0.66324 s
Kernel time -- : 0.62043 s
Speedup ------ : 1.06899
Efficiency --- : 0.53450
0.00222473 0.00249415 0.00283808 0.00286441 0.00346783 ... 4.99814 4.99854 4.99886 4.99908 4.99975 
0.00222473 0.00249415 0.00283808 0.00286441 0.00346783 ... 4.99814 4.99854 4.99886 4.99908 4.99975 
OK results :-)
*/
```
On parallélise les boucles d'initialisation et finalisation avec un simple *parallel for*. Pour la boucle de travail, on parallélise uniquement la boucle imbriquée pour éviter les problèmes de dépendances.

**Avec parallélisation**
```c
void enumeration_sort_kernel(double tab[N]) {
  size_t i, j;
  size_t* position = malloc(N * sizeof(size_t));
  double* copy     = malloc(N * sizeof(size_t));

  #pragma omp parallel for
  for (i = 0; i < N; i++) {
    position[i] = 0;
    copy[i] = tab[i];
  }
  

  for (j = 0; j < N; j++) {
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
      if ((tab[j] < tab[i]) || ((tab[i] == tab[j]) && (i < j))) {
        position[i]++;
      }
    }
  }

  #pragma omp parallel for
  for (i = 0; i < N; i++)
    tab[position[i]] = copy[i];

  free(position);
  free(copy);
}
/* Résultats :

Reference time : 0.67546 s
Kernel time -- : 0.24098 s
Speedup ------ : 2.80297
Efficiency --- : 1.40148
0.000156581 0.000615921 0.00190651 0.00208136 0.00234239 ... 4.99512 4.99565 4.99691 4.99718 4.99805 
0.000156581 0.000615921 0.00190651 0.00208136 0.00234239 ... 4.99512 4.99565 4.99691 4.99718 4.99805 
OK results :-)
*/
```

### 1.3 Tri à bulle

**Sans parallélisation**
```c
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
```
On parallélise la boucle for en **schedule(guided)** pour une meilleure **répartition du travail qui baisse quand i augmente**.
schedule(guided) est adaptée à la charge de travail ***"en triangle"***.

**Avec parallélisation**
```c
void bubble_sort_kernel(double tab[N]) {
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

Reference time : 2.13888 s
Kernel time -- : 1.90986 s
Speedup ------ : 1.11992
Efficiency --- : 0.55996
0.0214903 0.0382192 0.0431424 0.0460404 0.0666443 ... 499.805 499.82 499.887 499.951 499.972 
0.0214903 0.0382192 0.0431424 0.0460404 0.0666443 ... 499.805 499.82 499.887 499.951 499.972 
OK results :-)
*/
```

