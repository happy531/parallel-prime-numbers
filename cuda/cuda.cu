#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "numgen.c"
#include <stdbool.h>

__device__ bool is_prime_device(unsigned long int num) {
    if (num <= 1) return false;
    for (unsigned long int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void count_primes_parallel(unsigned long int *numbers, int *prime_count, int inputArgument) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < inputArgument) {
        if (is_prime_device(numbers[idx])) {
            atomicAdd(prime_count, 1); 
        }
    }
}

int main(int argc,char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);
    long inputArgument = ins__args.arg; 

    unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);  

    unsigned long int *dev_numbers;
    int *dev_prime_count;
    cudaMalloc(&dev_numbers, inputArgument * sizeof(unsigned long int));
    cudaMalloc(&dev_prime_count, sizeof(int));
    cudaMemcpy(dev_numbers, numbers, inputArgument * sizeof(unsigned long int), cudaMemcpyHostToDevice);
    cudaMemset(dev_prime_count, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (inputArgument + threadsPerBlock - 1) / threadsPerBlock; 
    count_primes_parallel<<<blocks, threadsPerBlock>>> (dev_numbers, dev_prime_count, inputArgument);

    int prime_count_host;
    cudaMemcpy(&prime_count_host, dev_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    printf("Number of prime numbers: %d\n", prime_count_host);

    cudaFree(dev_numbers);
    cudaFree(dev_prime_count); 
    free(numbers);

    return 0;
}
