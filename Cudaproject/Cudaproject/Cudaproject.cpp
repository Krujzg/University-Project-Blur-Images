#include <iostream>
#include "opencv2/opencv.hpp"
#include "..\cudakernel\kernel.h"
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace cv;

Mat imageInputRGBA;
Mat imageOutputRGBA;

uchar4* device_inputImageRGBA__;
uchar4* device_outputImageRGBA__;

float* host_filter__;

size_t numberOfRows() { return imageInputRGBA.rows; }
size_t numberOfColumns() { return imageInputRGBA.cols; }

void doBlur(const uchar4* const host_inputImageRGBA, uchar4* const device_inputImageRGBA, uchar4* const device_outputImageRGBA, const size_t numberOfRows, const size_t numberOfColumns,unsigned char* device_redBlurred,	unsigned char* device_greenBlurred,	unsigned char* device_blueBlurred,const int filterWidth);

void allocateAndCopy(const size_t numberOfRows, const size_t numberOfColumns, const float* const host_filter, const size_t filterWidth);

void freememory();

void start(uchar4** host_inputImageRGBA, uchar4** host_outputImageRGBA, uchar4** device_inputImageRGBA, uchar4** device_outputImageRGBA, unsigned char** device_redBlurred, unsigned char** device_greenBlurred,unsigned char** device_blueBlurred,float** host_filter, int* filterWidth)
{
	Mat image = imread("test2.jpg", IMREAD_ANYCOLOR);
	imshow("Original picture", image);
	waitKey();

	cvtColor(image, imageInputRGBA, COLOR_BGR2BGRA);

	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	*host_inputImageRGBA = (uchar4*)imageInputRGBA.ptr<unsigned char>(0);
	*host_outputImageRGBA = (uchar4*)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = numberOfRows() * numberOfColumns();

	cudaMalloc(device_inputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMalloc(device_outputImageRGBA, sizeof(uchar4) * numPixels);

	cudaMemcpy(*device_inputImageRGBA, *host_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

	device_inputImageRGBA__ = *device_inputImageRGBA;
	device_outputImageRGBA__ = *device_outputImageRGBA;





	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;

	*filterWidth = blurKernelWidth;

	*host_filter = new float[blurKernelWidth * blurKernelWidth];
	host_filter__ = *host_filter;

	//When converting the Gaussian’s continuous values into the discrete values needed for a kernel, the sum of the values will be different from 1. 
	//This will cause a darkening or brightening of the image. 
	//To remedy this, the values can be normalized by dividing each term in the kernel by the sum of all terms in the kernel. - Wikipédia
	float filterSum = 0.f; //normalizációhoz

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) 
	{
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) 
		{
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			(*host_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	/*for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) 
	{
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) 
		{
			(*host_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
		}
	}*/

	cudaMalloc(device_redBlurred, sizeof(unsigned char) * numPixels);
	cudaMalloc(device_greenBlurred, sizeof(unsigned char) * numPixels);
	cudaMalloc(device_blueBlurred, sizeof(unsigned char) * numPixels);
}

void end()
{
	const int numPixels = numberOfRows() * numberOfColumns();
	// visszamásolja az outputot a cpura
	cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), device_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

	Mat imageOutputBGR;
	cvtColor(imageOutputRGBA, imageOutputBGR, COLOR_BGR2BGRA);

	imshow("Blurred picture", imageOutputBGR);
	waitKey();

	cudaFree(device_inputImageRGBA__);
	cudaFree(device_outputImageRGBA__);
	delete[] host_filter__;
}

int main(int argc, char** argv) 
{
	uchar4* host_inputImageRGBA;
	uchar4* device_inputImageRGBA;
	uchar4* host_outputImageRGBA;
	uchar4* device_outputImageRGBA;
	unsigned char* device_redBlurred;
	unsigned char* device_greenBlurred;
	unsigned char* device_blueBlurred;

	float* host_filter;
	int    filterWidth;

	start(&host_inputImageRGBA,&host_outputImageRGBA,&device_inputImageRGBA,&device_outputImageRGBA,&device_redBlurred,&device_greenBlurred,&device_blueBlurred,&host_filter,&filterWidth );

	allocateAndCopy(numberOfRows(), numberOfColumns(), host_filter, filterWidth);

	doBlur(host_inputImageRGBA, device_inputImageRGBA, device_outputImageRGBA, numberOfRows(), numberOfColumns(), device_redBlurred, device_greenBlurred, device_blueBlurred, filterWidth);

	freememory();
	end();

	cudaFree(device_redBlurred);
	cudaFree(device_greenBlurred);
	cudaFree(device_blueBlurred);

	return 0;
}