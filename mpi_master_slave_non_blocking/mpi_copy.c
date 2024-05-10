#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "numgen.c"

#define TAG_WORK 1
#define TAG_RESULT 2

void print_array(unsigned long int *arr, size_t size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%lu", arr[i]);
        if (i != size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

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

void master(int comm_size, int rank, long inputArgument, unsigned long int *numbers) {
  int num_slaves = comm_size - 1;
  long work_per_slave = inputArgument / num_slaves;
  long remaining_work = inputArgument % num_slaves;

  MPI_Request *send_requests = malloc(num_slaves * sizeof(MPI_Request));

  for (int i = 1; i <= num_slaves; i++) {
    long start_index = (i - 1) * work_per_slave;
    long end_index = start_index + work_per_slave - 1;

    MPI_Isend(&work_per_slave, 1, MPI_LONG, i, TAG_WORK, MPI_COMM_WORLD, &send_requests[i-1]);
    MPI_Isend(&start_index, 1, MPI_LONG, i, TAG_WORK, MPI_COMM_WORLD, &send_requests[i-1]);
    MPI_Isend(&end_index, 1, MPI_LONG, i, TAG_WORK, MPI_COMM_WORLD, &send_requests[i-1]);
    MPI_Isend(numbers + start_index, work_per_slave, MPI_UNSIGNED_LONG, i, TAG_WORK, MPI_COMM_WORLD, &send_requests[i-1]);
  }

  if (remaining_work > 0) {
    int slave = num_slaves;
    long start_index = inputArgument - remaining_work;

    MPI_Isend(&remaining_work, 1, MPI_LONG, slave, TAG_WORK, MPI_COMM_WORLD, &send_requests[num_slaves - 1]);
    MPI_Isend(&start_index, 1, MPI_LONG, slave, TAG_WORK, MPI_COMM_WORLD, &send_requests[num_slaves - 1]);
    MPI_Isend(numbers + start_index, remaining_work, MPI_UNSIGNED_LONG, slave, TAG_WORK, MPI_COMM_WORLD, &send_requests[num_slaves - 1]);
  }

  int total_primes = 0;
  for (int i = 1; i <= num_slaves; i++) {
    int partial_count;
    MPI_Recv(&partial_count, 1, MPI_INT, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    total_primes += partial_count;

    MPI_Wait(&send_requests[i-1], MPI_STATUS_IGNORE); 
  }

  printf("Total prime numbers: %d\n", total_primes);
  free(send_requests);
}

void slave(int rank) {
  long work_size, start_index, end_index;
  MPI_Request recv_requests[4], send_request; 

  MPI_Irecv(&work_size, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &recv_requests[0]);
  MPI_Irecv(&start_index, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &recv_requests[1]);
  MPI_Irecv(&end_index, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &recv_requests[2]);

  unsigned long int *sub_numbers = (unsigned long int*)malloc(work_size * sizeof(unsigned long int));
  MPI_Irecv(sub_numbers, work_size, MPI_UNSIGNED_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &recv_requests[3]);

  MPI_Waitall(4, recv_requests, MPI_STATUSES_IGNORE); 

  int prime_count = 0;
  for (long i = start_index; i <= end_index; i++) {
    if (is_prime(sub_numbers[i - start_index])) { 
      prime_count++;
    }
  }

  MPI_Isend(&prime_count, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD, &send_request); // Non-blocking send
  free(sub_numbers); 

  MPI_Wait(&send_request, MPI_STATUS_IGNORE);  
}

int main(int argc,char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  long inputArgument = ins__args.arg; 

  struct timeval ins__tstart, ins__tstop;

  int myrank,nproc;
  unsigned long int *numbers;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  if(!myrank){
    gettimeofday(&ins__tstart, NULL);
	  numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  	numgen(inputArgument, numbers);
    // print_array(numbers, inputArgument);
  }

  // run your computations here (including MPI communication)

  if (!myrank) {
    master(nproc, myrank, inputArgument, numbers); 
  } else {
    slave(myrank);
  }

  // synchronize/finalize your computations

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }
  
  MPI_Finalize();

}
