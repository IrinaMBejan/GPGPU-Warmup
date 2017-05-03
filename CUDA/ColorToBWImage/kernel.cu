#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdio.h>

#include "utils.h"

__global__ void RGBAtoBW(const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	int numRows, int numCols)
{
	for (int i = 0; i < numRows; i++)
		for (int j = 0; j < numCols; j++)
		{
			uchar4 rgba = rgbaImage[i*numCols + j];
			float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
			greyImage[i*numCols + j] = channelSum;
		}
}

void cpuRGBAtoBW(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1); 
	RGBAtoBW <<<gridSize, blockSize >>>(d_rgbaImage, d_greyImage, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}