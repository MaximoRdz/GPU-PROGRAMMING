#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>

#include <iostream>

#include <cuda_runtime.h>
#include <mma.h>          // WMMA API
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../include/matmul_cuda.hpp"

using namespace nvcuda;

// WMMA tile geometry. 16x16x16 is the standard shape for half/bf16 on
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// How many warps (in x and y) cooperate per CUDA block. Each warp
// computes one WMMA_M x WMMA_N output tile.
#define WARPS_PER_BLOCK_X 4
#define WARPS_PER_BLOCK_Y 4

#define SHMEM_PAD 8

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t cerr = call;                                            \
        if (cerr != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(cerr));          \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------
// WMMA GEMM kernel.
//
// InT  : element type of A/B as they live in global memory
// AccT : accumulator type used inside the fragment and for the output
//        buffer (float, for both half and bf16 inputs).
//
// One warp computes one WMMA_M x WMMA_N tile of the output, looping
// over the K dimension in WMMA_K-sized steps. The CUDA block holds
// WARPS_PER_BLOCK_Y x WARPS_PER_BLOCK_X warps, i.e. it covers a
// (WARPS_PER_BLOCK_Y*WMMA_M) x (WARPS_PER_BLOCK_X*WMMA_N) region of
// the output per block
// just built out of warp-tiles instead of thread-tiles.
//
// A and B tiles are staged through shared memory (row-major, padded)
// so every warp in the block reuses the same loaded data instead of
// each warp re-reading global memory independently 
// ---------------------------------------------------------------------
template <typename InT, typename AccT>
__global__ void WmmaSqMatMulKernel(const InT* d_a, const InT* d_b,
        AccT* d_dst, int N)
{
    constexpr int BLOCK_ROWS = WARPS_PER_BLOCK_Y * WMMA_M;
    constexpr int BLOCK_COLS = WARPS_PER_BLOCK_X * WMMA_N;

    __shared__ InT tile_a[BLOCK_ROWS][WMMA_K + SHMEM_PAD];
    __shared__ InT tile_b[WMMA_K][BLOCK_COLS + SHMEM_PAD];

    // Which warp am I, and which output tile does that warp own?
    const int warpId    = threadIdx.x / warpSize;      // linear warp index within block
    const int warpRow   = warpId / WARPS_PER_BLOCK_X;   // warp's row within the block's warp grid
    const int warpCol    = warpId % WARPS_PER_BLOCK_X;   // warp's col within the block's warp grid

    const int blockRow0 = blockIdx.y * BLOCK_ROWS;      // top-left of this block's output region
    const int blockCol0 = blockIdx.x * BLOCK_COLS;

    const int tileRow0  = blockRow0 + warpRow * WMMA_M;  // top-left of this warp's output tile
    const int tileCol0  = blockCol0 + warpCol * WMMA_N;

    // Accumulator fragment lives in registers across the whole K loop.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccT> acc_frag;
    wmma::fill_fragment(acc_frag, AccT{});

    const int num_k_tiles = (N + WMMA_K - 1) / WMMA_K;

    // Flatten thread index within the block for the cooperative
    // global->shared loads below (every thread in the block helps
    // load, not just the owning warp -- same spirit as your existing
    // "each thread loads an element to the tile" comment).
    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int threadsPerBlk = blockDim.x * blockDim.y;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int kBase = t * WMMA_K;

        // --- cooperative load of the A strip: BLOCK_ROWS x WMMA_K ---
        for (int idx = tid; idx < BLOCK_ROWS * WMMA_K; idx += threadsPerBlk) {
            const int r = idx / WMMA_K;
            const int c = idx % WMMA_K;
            const int gRow = blockRow0 + r;
            const int gCol = kBase + c;
            tile_a[r][c] = (gRow < N && gCol < N) ? d_a[gRow * N + gCol] : InT{};
        }

        // --- cooperative load of the B strip: WMMA_K x BLOCK_COLS ---
        for (int idx = tid; idx < WMMA_K * BLOCK_COLS; idx += threadsPerBlk) {
            const int r = idx / BLOCK_COLS;
            const int c = idx % BLOCK_COLS;
            const int gRow = kBase + r;
            const int gCol = blockCol0 + c;
            tile_b[r][c] = (gRow < N && gCol < N) ? d_b[gRow * N + gCol] : InT{};
        }

        __syncthreads();

        // Each warp loads its own 16x16 sub-tile out of the shared
        // strips and issues the MMA instruction.
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InT, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InT, wmma::row_major> b_frag;

        const InT* a_ptr = &tile_a[warpRow * WMMA_M][0];
        const InT* b_ptr = &tile_b[0][warpCol * WMMA_N];

        wmma::load_matrix_sync(a_frag, a_ptr, WMMA_K + SHMEM_PAD);
        wmma::load_matrix_sync(b_frag, b_ptr, BLOCK_COLS + SHMEM_PAD);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads(); // before next tile overwrites shared memory
    }

    // Store this warp's tile back to global memory, bounds-checked.
    if (tileRow0 < N && tileCol0 < N) {
        wmma::store_matrix_sync(&d_dst[tileRow0 * N + tileCol0], acc_frag, N,
                                 wmma::mem_row_major);
    }
}

template <typename InT, typename AccT>
void LaunchWmmaSqMatMulKernel(const std::vector<InT>& h_a,
        const std::vector<InT>& h_b, std::vector<AccT>& h_dst)
{
    const int N = static_cast<int>(std::sqrt(h_a.size()));

    const size_t inBytes  = h_a.size() * sizeof(InT);
    const size_t outBytes = h_dst.size() * sizeof(AccT);

    InT*  d_a   = nullptr;
    InT*  d_b   = nullptr;
    AccT* d_dst = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_a, inBytes);     CUDA_CHECK(err);
    err = cudaMalloc(&d_b, inBytes);     CUDA_CHECK(err);
    err = cudaMalloc(&d_dst, outBytes);  CUDA_CHECK(err);

    err = cudaMemcpy(d_a, h_a.data(), inBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_b, h_b.data(), inBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemset(d_dst, 0, outBytes);
    CUDA_CHECK(err);

    // blockDim.x must be a multiple of warpSize (32); we pack
    // WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y warps into one block via
    // a flat x dimension, with blockDim.y left at 1. This is a common
    // WMMA convention -- warps are identified via threadIdx.x/32.
    dim3 blockDim(32 * WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y, 1);
    constexpr int BLOCK_ROWS = WARPS_PER_BLOCK_Y * WMMA_M;
    constexpr int BLOCK_COLS = WARPS_PER_BLOCK_X * WMMA_N;
    dim3 gridDim((N + BLOCK_COLS - 1) / BLOCK_COLS,
                 (N + BLOCK_ROWS - 1) / BLOCK_ROWS);

    WmmaSqMatMulKernel<InT, AccT><<<gridDim, blockDim>>>(d_a, d_b, d_dst, N);
    err = cudaGetLastError();
    CUDA_CHECK(err);
    CUDA_CHECK(cudaDeviceSynchronize());

    err = cudaMemcpy(h_dst.data(), d_dst, outBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_dst));
}

template void LaunchWmmaSqMatMulKernel<__half, float>(
        const std::vector<__half>& h_a,
        const std::vector<__half>& h_b,
        std::vector<float>& h_dst);

template void LaunchWmmaSqMatMulKernel<__nv_bfloat16, float>(
        const std::vector<__nv_bfloat16>& h_a,
        const std::vector<__nv_bfloat16>& h_b,
        std::vector<float>& h_dst);
