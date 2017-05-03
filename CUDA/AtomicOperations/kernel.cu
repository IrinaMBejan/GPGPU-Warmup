#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>

#define NUM_THREADS 1000000
#define ARRAY_SIZE  100

#define BLOCK_WIDTH 1000

__global__ void increment_atomic(int *g)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	atomicAdd(&g[i], 1);
}


void print_array(int *array, int size)
{
	printf("{ ");
	for (int i = 0; i < size; i++) { printf("%d ", array[i]); }
	printf("}\n");
}

int main(int argc, char **argv)
{
	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

	int h_array[ARRAY_SIZE];
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int * d_array;
	cudaMalloc((void **)&d_array, ARRAY_BYTES);
	cudaMemset((void *)d_array, 0, ARRAY_BYTES);

	increment_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_array);

	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	print_array(h_array, ARRAY_SIZE);

	cudaFree(d_array);
	return 0;
}