#include <cmath>
#include <vector>
#include <cstddef>

#include <iostream>

#include <cuda_runtime.h>

#include "../include/matmul_cuda.hpp"

#define BLOCK_DIM 16

#define CUDA_CHECK(call)       \
do {       \
    cudaError_t cerr = call;       \
    if (cerr != cudaSuccess) {       \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(cerr));       \
        exit(1);       \
    }       \
} while (0)       \


template <typename T>
__global__ void NaiveSqMatMulKernel(const T* d_a, const T* d_b,
        T* d_dst, int N)
{
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            d_dst[row * N + col] += d_a[row * N +  k] * d_b[k * N + col];
        }
    }
}


template <typename T>
void LaunchNaiveSqMatMulKernel(const std::vector<T>& h_a,
        const std::vector<T>& h_b, std::vector<T>& h_dst)
{
    /// matrix metadata
    const size_t matrixSizeBytes = h_a.size() * sizeof(T);
    const int N = static_cast<int>(std::sqrt(h_a.size()));
    
    /// initialize device data structures
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_dst = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_a, matrixSizeBytes);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_b, matrixSizeBytes);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_dst, matrixSizeBytes);
    CUDA_CHECK(err);

    /// transfer data from host to device
    err = cudaMemcpy(d_a, h_a.data(), matrixSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_b, h_b.data(), matrixSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    err = cudaMemset(d_dst, 0, matrixSizeBytes);
    CUDA_CHECK(err);

    /// set up GPU (blocks and threads)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // ceiling division.
            (N + blockDim.y - 1) / blockDim.y);

    /// launch kernel
    NaiveSqMatMulKernel<<<gridDim, blockDim>>>(d_a, d_b, d_dst, N);
    err = cudaGetLastError();
    CUDA_CHECK(err);

    CUDA_CHECK(cudaDeviceSynchronize());

    /// transfer results to host memory
    err = cudaMemcpy(h_dst.data(), d_dst, matrixSizeBytes,
            cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    /// free gpu memory
    err = cudaFree(d_a);
    CUDA_CHECK(err);

    err = cudaFree(d_b);
    CUDA_CHECK(err);

    err = cudaFree(d_dst);
    CUDA_CHECK(err);
}


template <typename T>
__global__ void OptimizedSqMatMulKernel(const T* d_a, const T* d_b,
        T* d_dst, int N)
{
    __shared__ T tile_a[BLOCK_DIM * BLOCK_DIM];
    __shared__ T tile_b[BLOCK_DIM * BLOCK_DIM];
    
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    const int num_tiles = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    T accumulator{};
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // each thread loads an element to the tile
        const int target_col = BLOCK_DIM * tile_id + threadIdx.x;
        const int target_row = BLOCK_DIM * tile_id + threadIdx.y;

        tile_a[threadIdx.y * BLOCK_DIM + threadIdx.x] = 
            (row < N) && (target_col < N) ?  d_a[row * N + target_col] :T{};

        tile_b[threadIdx.y * BLOCK_DIM + threadIdx.x] = 
            (target_row < N) && (col < N) ? d_b[target_row * N + col] : T{};

        __syncthreads(); // block threads synchronization


        for (int k = 0; k < BLOCK_DIM; ++k) {
            accumulator += 
                tile_a[threadIdx.y * BLOCK_DIM + k] *
                tile_b[k * BLOCK_DIM + threadIdx.x]; 
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        d_dst[row * N + col] = accumulator;
    }
}


template <typename T>
void LaunchOptimizedSqMatMulKernel(const std::vector<T>& h_a,
        const std::vector<T>& h_b, std::vector<T>& h_dst)
{
    /// matrix metadata
    const size_t matrixSizeBytes = h_a.size() * sizeof(T);
    const int N = static_cast<int>(std::sqrt(h_a.size()));
    
    /// initialize device data structures
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_dst = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_a, matrixSizeBytes);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_b, matrixSizeBytes);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_dst, matrixSizeBytes);
    CUDA_CHECK(err);

    /// transfer data from host to device
    err = cudaMemcpy(d_a, h_a.data(), matrixSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_b, h_b.data(), matrixSizeBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    err = cudaMemset(d_dst, 0, matrixSizeBytes);
    CUDA_CHECK(err);

    /// set up GPU (blocks and threads)
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // ceiling division.
            (N + blockDim.y - 1) / blockDim.y);

    /// launch kernel
    OptimizedSqMatMulKernel<<< gridDim, blockDim >>>(d_a, d_b, d_dst, N);
    err = cudaGetLastError();
    CUDA_CHECK(err);

    CUDA_CHECK(cudaDeviceSynchronize());

    /// transfer results to host memory
    err = cudaMemcpy(h_dst.data(), d_dst, matrixSizeBytes,
            cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    /// free gpu memory
    err = cudaFree(d_a);
    CUDA_CHECK(err);

    err = cudaFree(d_b);
    CUDA_CHECK(err);

    err = cudaFree(d_dst);
    CUDA_CHECK(err);
}

/// template instantiation (for general projecs just write the implementation
/// on .hpp or .tpp)

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/// Naive
    // integers
template void LaunchNaiveSqMatMulKernel<std::int8_t>(
        const std::vector<std::int8_t>& h_a,
        const std::vector<std::int8_t>& h_b, 
        std::vector<std::int8_t>& h_dst);

template void LaunchNaiveSqMatMulKernel<std::int16_t>(
        const std::vector<std::int16_t>& h_a,
        const std::vector<std::int16_t>& h_b, 
        std::vector<std::int16_t>& h_dst);

template void LaunchNaiveSqMatMulKernel<std::int32_t>(
        const std::vector<std::int32_t>& h_a,
        const std::vector<std::int32_t>& h_b, 
        std::vector<std::int32_t>& h_dst);

template void LaunchNaiveSqMatMulKernel<std::int64_t>(
        const std::vector<std::int64_t>& h_a,
        const std::vector<std::int64_t>& h_b, 
        std::vector<std::int64_t>& h_dst);

    // float
template void LaunchNaiveSqMatMulKernel<__nv_bfloat16>(
        const std::vector<__nv_bfloat16>& h_a,
        const std::vector<__nv_bfloat16>& h_b, 
        std::vector<__nv_bfloat16>& h_dst);

template void LaunchNaiveSqMatMulKernel<__half>(
        const std::vector<__half>& h_a,
        const std::vector<__half>& h_b, 
        std::vector<__half>& h_dst);

template void LaunchNaiveSqMatMulKernel<float>(
        const std::vector<float>& h_a,
        const std::vector<float>& h_b, 
        std::vector<float>& h_dst);

template void LaunchNaiveSqMatMulKernel<double>(
        const std::vector<double>& h_a,
        const std::vector<double>& h_b, 
        std::vector<double>& h_dst);

/// Optimized Tiling
    // integers
template void LaunchOptimizedSqMatMulKernel<std::int8_t>(
        const std::vector<std::int8_t>& h_a,
        const std::vector<std::int8_t>& h_b, 
        std::vector<std::int8_t>& h_dst);

template void LaunchOptimizedSqMatMulKernel<std::int16_t>(
        const std::vector<std::int16_t>& h_a,
        const std::vector<std::int16_t>& h_b, 
        std::vector<std::int16_t>& h_dst);

template void LaunchOptimizedSqMatMulKernel<std::int32_t>(
        const std::vector<std::int32_t>& h_a,
        const std::vector<std::int32_t>& h_b, 
        std::vector<std::int32_t>& h_dst);

template void LaunchOptimizedSqMatMulKernel<std::int64_t>(
        const std::vector<std::int64_t>& h_a,
        const std::vector<std::int64_t>& h_b, 
        std::vector<std::int64_t>& h_dst);

    // float
template void LaunchOptimizedSqMatMulKernel<__nv_bfloat16>(
        const std::vector<__nv_bfloat16>& h_a,
        const std::vector<__nv_bfloat16>& h_b, 
        std::vector<__nv_bfloat16>& h_dst);

template void LaunchOptimizedSqMatMulKernel<__half>(
        const std::vector<__half>& h_a,
        const std::vector<__half>& h_b, 
        std::vector<__half>& h_dst);

template void LaunchOptimizedSqMatMulKernel<float>(
        const std::vector<float>& h_a,
        const std::vector<float>& h_b, 
        std::vector<float>& h_dst);

template void LaunchOptimizedSqMatMulKernel<double>(
        const std::vector<double>& h_a,
        const std::vector<double>& h_b, 
        std::vector<double>& h_dst);

