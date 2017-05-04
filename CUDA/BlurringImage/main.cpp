#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
	uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
	unsigned char **d_redBlurred,
	unsigned char **d_greenBlurred,
	unsigned char **d_blueBlurred,
	float **h_filter, int *filterWidth,
	const std::string &filename);

void postProcess(const std::string& output_file);

void gaussianBlur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA,
	const size_t numRows, const size_t numCols,
	unsigned char *d_redBlurred,
	unsigned char *d_greenBlurred,
	unsigned char *d_blueBlurred,
	const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
	const float* const h_filter, const size_t filterWidth);

void cleanup();

int main(int argc, char **argv) 
{
	uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

	float *h_filter;
	int    filterWidth;

	std::string input_file = "C:\\Users\\irina\\Desktop\\RGBAPicture.jpg";
	std::string output_file = "HW1_output.jpg";
	
	preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
		&d_redBlurred, &d_greenBlurred, &d_blueBlurred,
		&h_filter, &filterWidth, input_file);

	allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
	
	GpuTimer timer;
	timer.Start();
	gaussianBlur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
		d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
	timer.Stop();
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	printf("%f msecs.\n", timer.Elapsed());

	cleanup();
	postProcess(output_file);

	checkCudaErrors(cudaFree(d_redBlurred));
	checkCudaErrors(cudaFree(d_greenBlurred));
	checkCudaErrors(cudaFree(d_blueBlurred));

	return 0;
}


void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
	uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
	unsigned char **d_redBlurred,
	unsigned char **d_greenBlurred,
	unsigned char **d_blueBlurred,
	float **h_filter, int *filterWidth,
	const std::string &filename) 
{
	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	
	if (image.empty()) 
	{
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	*h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4)));

	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImageRGBA__ = *d_inputImageRGBA;
	d_outputImageRGBA__ = *d_outputImageRGBA;

	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;

	*filterWidth = blurKernelWidth;
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;

	float filterSum = 0.f;

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) 
	{
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) 
		{
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) 
	{
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) 
		{
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
		}
	}

	checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels));

	checkCudaErrors(cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file) 
{
	const int numPixels = numRows() * numCols();

	checkCudaErrors(cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	cv::Mat imageOutputBGR;
	cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);

	cv::imwrite(output_file.c_str(), imageOutputBGR);

	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	delete[] h_filter__;
}
