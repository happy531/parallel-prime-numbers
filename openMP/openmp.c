#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "numgen.c"

int is_prime(unsigned long int num) {
  if (num <= 1) return 0; 
  if (num <= 3) return 1;
  if (num % 2 == 0 || num % 3 == 0) return 0; 

  for (unsigned long int i = 5; i * i <= num; i += 6) {
      if (num % i == 0 || num % (i + 2) == 0) {
          return 0;
      }
  }

  return 1; 
}

int main(int argc,char **argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  //set number of threads
  omp_set_num_threads(ins__args.n_thr);
  
  //program input argument
  long inputArgument = ins__args.arg; 
  unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  
  // run your computations here (including OpenMP stuff)

  int prime_count = 0;

  #pragma omp parallel for reduction(+:prime_count)
  for (long i = 0; i < inputArgument; i += 1000) {
      for (long j = i; j < i + 1000 && j < inputArgument; j++) {
        if (is_prime(numbers[j])) {
          prime_count++;
        }
      }
  }


  printf("Number of primes found: %d\n", prime_count); 

  // synchronize/finalize your computations
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
}
