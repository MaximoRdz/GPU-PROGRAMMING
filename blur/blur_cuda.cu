/* shared memory optimization: both vertical and horizontal kernels.*/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "blur_cuda.hpp"

#define MAX_KERNEL_SIZE 512

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

extern __shared__ unsigned char tile_horizontal[];
extern __shared__ unsigned char tile_vertical[];

__global__ void Convolve1DHorizontalKernel(const unsigned char* d_src, unsigned char* d_dst,
        int ncols, int nrows, size_t kernel_size,
        int kernel_radius)
{
    /// cooperative loading into shared memory
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int tile_width = blockDim.x + 2 * kernel_radius;
    const int block_row_offset = threadIdx.y * tile_width;
    const int block_col_start = blockIdx.x * blockDim.x;
    
    // main pixel value 
    if (col < ncols && row < nrows) {
        // (first tile pixel we do plus kernel_radius to offset away from the halo)
        tile_horizontal[block_row_offset + threadIdx.x + kernel_radius] = d_src[row * ncols + col];
    }
    // halo loading 
    // strided: correct for kernel_radius > blockDim.x which might happen for large sigmas
    if (row < nrows){
        for (int i = threadIdx.x; i < kernel_radius; i += blockDim.x) {
            // left halo: [0, kernel_radius)
            const int left_col = block_col_start - kernel_radius + i;
            tile_horizontal[block_row_offset + i] = 
                (left_col >= 0) ? d_src[row * ncols + left_col] : 0;
                // if desired mirror edges, periodic, etc. could be implemented here

            // right halo: [blockDim.x + kernel_radius, blockDim.x + 2*kernel_radius)
            const int right_col = block_col_start + blockDim.x + i;
            tile_horizontal[block_row_offset + blockDim.x + kernel_radius + i] = 
                (right_col < ncols) ? d_src[row * ncols + right_col] : 0;
        }
    }

    __syncthreads(); // every thread of the block must reach this point before any of them continues

    /// convolution using shared memory
    if (col < ncols && row < nrows) {
        float accumulator = 0.0;

        for (int k = 0; k < kernel_size; ++k){
            const int offset = k - kernel_radius;
            const int target_col = threadIdx.x + kernel_radius + offset;

            accumulator += c_kernel[k] * tile_horizontal[block_row_offset + target_col];
        }

    d_dst[row * ncols + col] = accumulator;
    }
}

__global__ void Convolve1DVerticalKernel(const unsigned char* d_src, unsigned char* d_dst,
        int ncols, int nrows, size_t kernel_size,
        int kernel_radius)
{
    /// cooperative loading into shared memory
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    /* const int tile_height = blockDim.y + 2 * kernel_radius; */
    const int tile_width = blockDim.x;
    const int block_row_start = blockIdx.y * blockDim.y;

    // main pixel value
    if (col < ncols && row < nrows) {
        tile_vertical[(threadIdx.y + kernel_radius) * tile_width + threadIdx.x] = 
            d_src[row * ncols + col];
    }
    // halo loading
    if (col < ncols) {
        // strided kernel_radius > blockdim.y (might happen)
        for (int i = threadIdx.y; i < kernel_radius; i += blockDim.y) {
            // upper halo [0, kernel_radius)
            const int upper_row = block_row_start - kernel_radius + i;
            tile_vertical[i * tile_width + threadIdx.x] = 
                (upper_row >= 0) ? d_src[upper_row * ncols + col] : 0;

            const int bottom_row = block_row_start + blockDim.y + i;
            tile_vertical[(blockDim.y + kernel_radius + i) * tile_width + threadIdx.x] = 
                (bottom_row < nrows) ? d_src[bottom_row * ncols + col] : 0;
        }
    }

    __syncthreads();

    if (col < ncols && row < nrows) {
        float accumulator = 0.0;

        for (int k = 0; k < kernel_size; ++k){
            /* const int offset = k - kernel_radius; */
            const int target_row = threadIdx.y + k;

            accumulator += 
                c_kernel[k] * tile_vertical[target_row * tile_width + threadIdx.x];
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


    /// c_kernel now lives in GPU constant memory
    CUDA_CHECK(
            cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size * sizeof(float))
            );
    
    /// GPU Threads Set UP
    dim3 blockDim(16, 16); // 16 threads x 16 threads

    // shared memory:
    size_t shared_bytes_horizontal =
        blockDim.y * (blockDim.x + 2 * kernel_radius) * sizeof(unsigned char);
    size_t shared_bytes_vertical = 
        (blockDim.y + 2 * kernel_radius) * blockDim.x * sizeof(unsigned char);

    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x, 
            (nrows + blockDim.y - 1) / blockDim.y);

    Convolve1DHorizontalKernel<<<
        gridDim, 
        blockDim,
        shared_bytes_horizontal        // each block gets its own shared memory tile+halo array
    >>>(d_src, d_buffer, ncols, nrows, kernel_size, kernel_radius);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Convolve1DVerticalKernel<<<
        gridDim,
        blockDim,
        shared_bytes_vertical
    >>>(d_buffer, d_dst, ncols, nrows, kernel_size, kernel_radius);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
            cudaMemcpy(h_dst, d_dst, array_size, cudaMemcpyDeviceToHost)
            );

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_buffer));
}
