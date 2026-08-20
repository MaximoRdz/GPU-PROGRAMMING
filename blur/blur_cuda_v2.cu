/* constant-memory array for kernel repeated access optimization */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "blur_cpu.hpp"
#include "blur_cuda.hpp"

#define MAX_KERNEL_SIZE 64

#define CUDA_CHECK(call)       \
do {       \
    cudaError_t err = call;       \
    if (err != cudaSuccess) {       \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);       \
    }       \
} while (0)       \


__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void Convolve1DHorizontalKernel(const unsigned char* d_src, unsigned char* d_dst,
        int ncols, int nrows, size_t kernel_size,
        int kernel_radius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < ncols && row < nrows) {
        int offset, target_col;
        float accumulator = 0.0;

        for (int k = 0; k < kernel_size; ++k){
            offset = k - kernel_radius;
            target_col = col + offset;

            if (!(target_col >= 0 && target_col < ncols)) continue;

            accumulator += c_kernel[k] * d_src[row * ncols + target_col];
        }

    d_dst[row * ncols + col] = accumulator;
    }
}

__global__ void Convolve1DVerticalKernel(const unsigned char* d_src, unsigned char* d_dst,
        int ncols, int nrows, size_t kernel_size,
        int kernel_radius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < ncols && row < nrows) {
        int offset, target_row;
        float accumulator = 0.0;

        for (int k = 0; k < kernel_size; ++k){
            offset = k - kernel_radius;
            target_row = row + offset;

            if (!(target_row >= 0 && target_row < nrows)) continue;

            accumulator += c_kernel[k] * d_src[target_row * ncols + col];
        }

    d_dst[row * ncols + col] = accumulator;
    }
}


void LaunchGaussianSmoothing(const unsigned char* h_src, unsigned char* h_dst,
        int ncols, int nrows, float* h_kernel, size_t kernel_size)
{
    const int kernel_radius = static_cast<int>(kernel_size / 2);
    
    size_t array_size = static_cast<size_t>(ncols * nrows) * sizeof(unsigned char);

    unsigned char *d_src = nullptr, *d_dst = nullptr, *d_buffer= nullptr; 

    CUDA_CHECK(cudaMalloc(&d_src, array_size));
    CUDA_CHECK(cudaMalloc(&d_dst, array_size));
    CUDA_CHECK(cudaMalloc(&d_buffer, array_size));

    CUDA_CHECK(
            cudaMemcpy(d_src, h_src, array_size, cudaMemcpyHostToDevice)
            );


    CUDA_CHECK(
            cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size * sizeof(float))
            );
    
    /// c_kernel now lives in GPU constant memory


    dim3 blockDim(16, 16); // 256 blocks
    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x, 
            (nrows + blockDim.y - 1) / blockDim.y);

    Convolve1DHorizontalKernel<<<gridDim, blockDim>>>(d_src, d_buffer, ncols, nrows, kernel_size,
            kernel_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Convolve1DVerticalKernel<<<gridDim, blockDim>>>(d_buffer, d_dst, ncols, nrows, kernel_size,
            kernel_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
            cudaMemcpy(h_dst, d_dst, array_size, cudaMemcpyDeviceToHost)
            );

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_buffer));
}
