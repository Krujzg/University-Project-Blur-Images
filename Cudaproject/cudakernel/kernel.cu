#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

#include <stdio.h>

__global__
void gausBlur(const unsigned char* const inputChannel,	unsigned char* const outputChannel,	int numberOfRows, int numberOfColumns,const float* const filter, const int filterWidth)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= numberOfColumns || py >= numberOfRows) {return;}

	float c = 0.0f;

	for (int fx = 0; fx < filterWidth; fx++) {
		for (int fy = 0; fy < filterWidth; fy++) {
			int imagex = px + fx - filterWidth / 2;
			int imagey = py + fy - filterWidth / 2;

			int maxX = imagex > 0 ? imagex : 0;
			int minX = maxX < numberOfColumns - 1 ? maxX : numberOfColumns - 1;
			imagex = minX;

			int maxY = imagey > 0 ? imagey : 0;
			int minY = maxY < numberOfColumns - 1 ? maxY : numberOfColumns - 1;
			imagey = minY;

			c += (filter[fy * filterWidth + fx] * inputChannel[imagey * numberOfColumns + imagex]);
		}
	}

	outputChannel[py * numberOfColumns + px] = c;
}
__global__
void separateTheChannels(const uchar4* const inputImageRGBA,int numberOfRows,int numberOfColumns,unsigned char* const redChannel,unsigned char* const greenChannel,unsigned char* const blueChannel)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;

	if (px >= numberOfColumns || py >= numberOfRows) {return;}

	int i = py * numberOfColumns + px;
	redChannel[i] = inputImageRGBA[i].x;
	greenChannel[i] = inputImageRGBA[i].y;
	blueChannel[i] = inputImageRGBA[i].z;
}

__global__
void putTogetherTheChannels(const unsigned char* const redChannel,const unsigned char* const greenChannel,const unsigned char* const blueChannel,uchar4* const outputImageRGBA,int numberOfRows,int numberOfColumns)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,	blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numberOfColumns + thread_2D_pos.x;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//255 -> átlátszóság megakadályozása
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* device_red;
unsigned char* device_green;
unsigned char* device_blue;

float* device_filter;

void allocateAndCopy(const size_t numberOfRows, const size_t numberOfColumns,const float* const host_filter, const size_t filterWidth)
{
	cudaMalloc(&device_red, sizeof(unsigned char) * numberOfRows * numberOfColumns);
	cudaMalloc(&device_green, sizeof(unsigned char) * numberOfRows * numberOfColumns);
	cudaMalloc(&device_blue, sizeof(unsigned char) * numberOfRows * numberOfColumns);

	cudaMalloc(&device_filter, sizeof(float) * filterWidth * filterWidth);

	cudaMemcpy(device_filter, host_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void doBlur(const uchar4* const host_inputImageRGBA, uchar4* const device_inputImageRGBA,uchar4* const device_outputImageRGBA, const size_t numberOfRows, const size_t numberOfColumns,unsigned char* device_redBlurred,unsigned char* device_greenBlurred,	unsigned char* device_blueBlurred,const int filterWidth)
{
	const dim3 blockSize(16, 16, 1);

	const dim3 gridSize(numberOfColumns / blockSize.x + 1, numberOfRows / blockSize.y + 1, 1);

	separateTheChannels <<<gridSize, blockSize >> > (device_inputImageRGBA, numberOfRows, numberOfColumns, device_red, device_green, device_blue);

	gausBlur <<<gridSize, blockSize >> > (device_red, device_redBlurred, numberOfRows, numberOfColumns, device_filter, filterWidth);
	gausBlur <<<gridSize, blockSize >> > (device_green, device_greenBlurred, numberOfRows, numberOfColumns, device_filter, filterWidth);
	gausBlur <<<gridSize, blockSize >> > (device_blue, device_blueBlurred, numberOfRows, numberOfColumns, device_filter, filterWidth);

	putTogetherTheChannels <<<gridSize, blockSize >> > (device_redBlurred, device_greenBlurred, device_blueBlurred, device_outputImageRGBA, numberOfRows, numberOfColumns);
}

void freememory()
{
	cudaFree(device_red);
	cudaFree(device_green);
	cudaFree(device_blue);
}