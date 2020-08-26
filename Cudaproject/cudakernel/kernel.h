#pragma once
#include <iostream>
#include "..\cudakernel\kernel.h"
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void allocateAndCopy(const size_t numberOfRows, const size_t numberOfColumns,const float* const host_filter, const size_t filterWidth);

void doBlur(const uchar4* const host_inputImageRGBA, uchar4* const device_inputImageRGBA,uchar4* const device_outputImageRGBA, const size_t numberOfRows, const size_t numberOfColumns,unsigned char* device_redBlurred,unsigned char* device_greenBlurred,	unsigned char* device_blueBlurred,	const int filterWidth);

void freememory();