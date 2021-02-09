# Compte rendu TP1 Programmation Parallèle

## 1) Echauffement
### 1.1 Régions parallèles OpenMP

Sans l'option -fopenmp , le programme affiche :
```
Hello
World
```
Avec l'option, le programme affiche :
```
Hello
Hello
Hello
Hello
World
```
Il y a 4 threads par défaut sur ma machine

Pour changer la valeur de OMP_NUM_THREADS : 
```
export OMP_NUM_THREADS=8
```
Pour modifier le nombre de threads statiquement : 
```c
#pragma omp parallel num_threads(2)
```
Le mot World ne s'affichera pas avant Hello car le *printf("World\n")* n'est pas éxécuté en parallèle et le  *'\n'* force l'affichage du printf. 

Pour que chaque thread éxécute les deux printf, il faut rajouter les accolades autour des deux printf : 
```c
#include <stdio.h>
#include <omp.h>

int main() {
  #pragma omp parallel num_threads(2)
  {
    printf("Hello\n");
    printf("World\n");
  }
  return 0;
}
```
### 1.2 Partage de travail OpenMP
Ce programme aditionne les vecteurs A et B dans le vecteur C
Quels instructions sont exécutées par un ou tous les threads :
```c
#include <stdio.h>
#include <omp.h>
#define SIZE  100
#define CHUNK  10

int main() {
  int i, tid;						  // 1 thread (main)
  double a[SIZE], b[SIZE], c[SIZE], sum = 0.;		  // 1 thread (main)
  
  for (i = 0; i < SIZE; i++)				  // 1 thread (main)
    a[i] = b[i] = i;					  // 1 thread (main)

  #pragma omp parallel private(tid) reduction(+: sum)
  {
    tid = omp_get_thread_num();				  // Tous les threads
    if (tid == 0)					  // Tous les threads
      printf("Nb threads = %d\n", omp_get_num_threads()); // 1 thread
    printf("Thread %d: starting...\n", tid);		  // Tous les threads

    #pragma omp for
    for (i = 0; i < SIZE; i++) {			  // Les i itérations de la 
      c[i] = a[i] + b[i];				  // boucle for sont partagées
      sum += c[i];					  // entre les threads donc 
      printf("Thread %d: c[%2d] = %g\n", tid, i, c[i]);   // 1 thread exécute ses itérations
    }
  }
  printf("sum = %g\n", sum);				  // 1 thread (main)
  return 0;						  // 1 thread (main)
}
```
La directive **for** indique que les itérations de la boucle qui suit doivent être partagées entre les membres de l'équipe de threads.

Quel est le statut des variables i et tid? Quel est le statut des variables a, b et c?
**Au début i, tid, a, b, c sont** ***shared***
**Ensuite,** *#pragma omp parallel private(tid) reduction(+: sum)* rend **tid** ***private*** et *#pragma omp for* rend **i** ***private***

La clause reduction crée un copie privée des variable listées pour chaque thread. 
A la fin, rassemble les copies avec l'opérateur précisé (ici : +)

## 2 Parallélisation de codes avec OpenMP
### 2.1 Directives OpenMP
#### 2.1.1 Addition de vecteurs

**Sans parallélisation**
```c
void addvec_kernel(double c[N], double a[N], double b[N]) {
  
  for (size_t i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}
/* Résultat :
Reference time : 0.03099 s
Kernel time -- : 0.03327 s
Speedup ------ : 0.93148
Efficiency --- : 0.46574
OK results :-)
*/
```
On parallélise la boucle for (en testant avec 8 threads)

**Avec parallélisation**
```c
void addvec_kernel(double c[N], double a[N], double b[N]) {
  #pragma omp parallel num_threads(8)
  {
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
  }
}
/* Résultat :
Reference time : 0.03040 s
Kernel time -- : 0.01658 s
Speedup ------ : 1.83380
Efficiency --- : 0.91690
OK results :-)
*/
```
### 2.2 Somme des éléments d'un vecteur

**Sans parallélisation**
```c
void sum_kernel(double* psum, double a[N]) {
  double sum = 0.;

  for (size_t i = 0; i < N; i++) {
    sum += a[i];
  }
  *psum = sum;
}
/* Résultats :
Reference time : 0.01637 s
Kernel time    : 0.01694 s
Speedup ------ : 0.96660
Efficiency --- : 0.48330
OK results :-)
*/
```
On parallélise la boucle for et on crée des copies privées de la variable *sum* pour chaque thread (copies réassemblées en somme à la fin).

**Avec parallélisation**
```c
void sum_kernel(double* psum, double a[N]) {
  double sum = 0.;
  #pragma omp parallel reduction(+: sum)
  {
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
      sum += a[i];
    }
  }
  *psum = sum;
}
/* Résultats :
Reference time : 0.01614 s
Kernel time    : 0.00640 s
Speedup ------ : 2.52106
Efficiency --- : 1.26053
OK results :-)
*/
```
### 2.3 Produit matrice-vecteur

**Sans parallélisation**
```c
void matvec_reference(double c[N], double A[N][N], double b[N]) {
  size_t i, j;

  for (i = 0; i < N; i++) {
    c[i] = 0.;
    for (j = 0; j < N; j++) {
      c[i] += A[i][j] * b[j];
    }
  }
}
/* Résultats :
Reference time : 0.09928 s
Kernel time -- : 0.13429 s
Speedup ------ : 0.73927
Efficiency --- : 0.36964
OK results :-)
*/
```
Il faut privatiser j! Sinon l'incrémentation (j++ dans le for) pourrait se faire de manière anarchique

**Avec parallélisation**
```c
void matvec_kernel(double c[N], double A[N][N], double b[N]) {
  size_t i, j;

  #pragma omp parallel
  {
    #pragma omp for private(j)
    for (i = 0; i < N; i++) {
      c[i] = 0.;
      for (j = 0; j < N; j++) {
        c[i] += A[i][j] * b[j];
      }
    }
  }
}
/* Résultats :
Reference time : 0.13929 s
Kernel time -- : 0.08742 s
Speedup ------ : 1.59341
Efficiency --- : 0.79670
OK results :-)
*/
```

## 3 Restructuration de programmes
### 3.1 Stencil 1D

Avec une parallélisation **for**, il y aura un problème quand un thread voudra utiliser une valeur qui n'est pas prévue dans ses itérations. En particulier la première itération de son chunk quand il voudra accéder à l'indice **i - 1**.
**Sans parallélisation**
```c
void stencil1D_kernel(double a[N], double b[N]) {
  for (size_t i = 1; i < N; i++) {
    a[i] = (a[i] + a[i - 1]) / 2;
    b[i] = (b[i] + b[i - 1]) / 2;
  }
}
/* Résultats :
Reference time : 0.48829 s
Kernel time -- : 0.47236 s
Speedup ------ : 1.03372
Efficiency --- : 0.51686
OK results :-)
*/
```
En l'état la boucle n'est pas parallélisable à cause de dépendances. J'ai donc essayé de paralléliser au niveau de a et b en séparant distribuant a et b en 2 boucles exécutées en parallèle par 2 threads (chacun sa boucle normalement).

**Avec parallélisation**
```c
// Computation kernel (to parallelize)
void stencil1D_kernel(double a[N], double b[N]) {
  
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single nowait
    {
      for (size_t i = 1; i < N; i++) {
        a[i] = (a[i] + a[i - 1]) / 2;
      }
    }

    #pragma omp single
    {
      for (size_t i = 1; i < N; i++) {
        b[i] = (b[i] + b[i - 1]) / 2;
      }
    }

  }
}
/* Résultats :
Reference time : 0.05051 s
Kernel time -- : 0.04415 s
Speedup ------ : 1.14388
Efficiency --- : 0.57194
OK results :-)
*/
```

### 3.2 Réduction et réinitialisation
Pas de résultat concluant

### 3.3 Multiplication de polynômes
Pas de résultat concluant
