#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void cpuRGBAtoBW(const uchar4 * const h_rgbaImage,
	uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage,
	size_t numRows, size_t numCols);

void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string& filename);

void postProcess(const std::string& output_file);

int main(int argc, char **argv)
{
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	std::string input_file = "C:\\Users\\irina\\Desktop\\RGBAPicture.jpg";
	std::string output_file = "HW1_output.jpg";
	
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	cpuRGBAtoBW(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	postProcess(output_file);

	return 0;
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename)
{
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	imageGrey.create(image.rows, image.cols, CV_8UC1);

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file)
{
	const int numPixels = numRows() * numCols();

	checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) *numPixels, cudaMemcpyDeviceToHost));

	cv::imwrite(output_file.c_str(), imageGrey);

	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}
