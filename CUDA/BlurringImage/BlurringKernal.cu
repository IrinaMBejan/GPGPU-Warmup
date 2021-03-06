#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdio.h>

#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= numCols || y >= numRows)
		return;

	int position = y*numCols + x;

	float color = 0.0f;

	for(int i = 0; i < filterWidth; i++)
		for (int j = 0; j < filterWidth; j++)
		{
			int xnew = x + j - filterWidth / 2;
			int ynew = y + i - filterWidth / 2;

			xnew = min(max(xnew, 0), numCols - 1);
			ynew = min(max(ynew, 0), numRows - 1);

			float value = filter[ynew * filterWidth + xnew];
			color += value * (float)inputChannel[ynew*numCols + xnew];
		}

	outputChannel[position] = color;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= numCols || y >= numRows)
		return;

	int position = y*numCols + x;

	redChannel[position] = inputImageRGBA[position].x;
	greenChannel[position] = inputImageRGBA[position].y;
	blueChannel[position] = inputImageRGBA[position].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
	const float* const h_filter, const size_t filterWidth)
{
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)* filterWidth * filterWidth));

	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void gaussianBlur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
	unsigned char *d_redBlurred,
	unsigned char *d_greenBlurred,
	unsigned char *d_blueBlurred,
	const int filterWidth)
{
	const dim3 blockSize(32, 32);

	const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);

	separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//call gaussian blur fo each color channel
	gaussian_blur << <gridSize, blockSize >> > (
		d_blue,
		d_blueBlurred,
		numRows,
		numCols,
		d_filter,
		filterWidth
		);

	gaussian_blur << <gridSize, blockSize >> > (
		d_red,
		d_redBlurred,
		numRows,
		numCols,
		d_filter,
		filterWidth
		);
	
	gaussian_blur << <gridSize, blockSize >> > (
		d_green,
		d_greenBlurred,
		numRows,
		numCols,
		d_filter,
		filterWidth
		);
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	recombineChannels << <gridSize, blockSize >> >(d_redBlurred,
		d_greenBlurred,
		d_blueBlurred,
		d_outputImageRGBA,
		numRows,
		numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
}
