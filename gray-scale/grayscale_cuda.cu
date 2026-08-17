#include "grayscale_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)       \
do {       \
    cudaError_t err = call;       \
    if (err != cudaSuccess) {       \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);       \
    }       \
} while (0)       \

// __global__: function call from the CPU but run on the GPU
__global__ void grayScaleKernel(
        const unsigned char* input, unsigned char* output,
        int width, int height, int channels
        )
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * channels;
        unsigned char B = input[idx + 0];
        unsigned char R = input[idx + 1];
        unsigned char G = input[idx + 2];

        output[row * width + col] =
            (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B);
    }
}

void launchGrayScaleConversion(const unsigned char* h_input, 
        unsigned char* h_output, int width, int height, int channels, 
        float* kernelTimeMs, float* totalTimeMs)
{
    size_t inputSize = (size_t)(width) * height * channels;
    size_t outputSize = (size_t)(width) * height;

    unsigned char *d_input = nullptr, *d_output = nullptr;

    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    CUDA_CHECK(cudaEventRecord(startTotal));

    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    CUDA_CHECK(
                cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice)
            );

    dim3 blockDim(16, 16); // 256 blocks
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
            (height + blockDim.y - 1) / blockDim.y);

    CUDA_CHECK(cudaEventRecord(startKernel));

    grayScaleKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);

    CUDA_CHECK(cudaEventRecord(stopKernel));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
}

