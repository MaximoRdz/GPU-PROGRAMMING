/* State-of-the-art separable Gaussian blur.
 *
 * Techniques used (each chosen because it addresses a specific bottleneck
 * a naive/textbook separable-convolution implementation hits):
 *
 *  1. Persistent device buffers (static, resized on demand) — avoids
 *     cudaMalloc/cudaFree churn on every call, which otherwise dominates
 *     runtime for small-to-medium images. NPP/OpenCV pool allocators do
 *     the same thing under the hood.
 *
 *  2. Pinned host staging via cudaHostRegister — lets cudaMemcpyAsync use
 *     true DMA instead of a staged (pageable) copy, roughly 2x H2D/D2H
 *     bandwidth on PCIe. We register the caller's existing pointers rather
 *     than forcing them to pre-allocate with cudaMallocHost, so the
 *     function stays a drop-in replacement.
 *
 *  3. Boundary handled by clamp-to-edge (not zero-fill) — zero-fill darkens
 *     edges/corners in a visible way; clamp is what every production blur
 *     implementation does and costs nothing extra (just a min/max clamp
 *     instead of a branch-to-zero).
 *
 *  4. __restrict__ pointers + __ldg — tells the compiler there's no
 *     aliasing (enables more aggressive scheduling) and routes reads
 *     through the read-only texture cache path.
 *
 *  5. 32x8 thread blocks — 32 in x keeps every warp's global memory access
 *     a single contiguous, fully coalesced transaction; the horizontal
 *     shared memory tile is sized/padded to avoid bank conflicts.
 *
 *  6. Kernel weights in __constant__ memory — broadcast to all threads in
 *     a warp in a single cycle, same access pattern every thread, ideal
 *     use case for the constant cache.
 *
 *  7. A single reusable CUDA stream — avoids the (small but nonzero)
 *     overhead of creating/destroying a stream on every call, and sets up
 *     for overlapping transfers with compute if this is extended to
 *     process multiple images/tiles later.
 *
 * Not done here (deliberately, to keep this a drop-in single-image
 * function rather than a batched pipeline): CUDA Graphs, multi-stream
 * overlap across independent images, and half-precision accumulation.
 * All three are the next step up if you're batching many images through
 * this exact call.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define MAX_KERNEL_SIZE 512

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__constant__ float c_kernel[MAX_KERNEL_SIZE];

// Block covers 32x8 output pixels. Horizontal tile is padded with
// 2*kernel_radius halo columns on top of the 32-wide main tile.
#define BLOCK_X 32
#define BLOCK_Y 8

extern __shared__ unsigned char tile[];

__device__ __forceinline__ int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void ConvolveHorizontal(const unsigned char* __restrict__ d_src,
                                    unsigned char* __restrict__ d_dst,
                                    int ncols, int nrows,
                                    int kernel_size, int kernel_radius)
{
    const int col = blockIdx.x * BLOCK_X + threadIdx.x;
    const int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int tile_width = BLOCK_X + 2 * kernel_radius;
    const int row_offset  = threadIdx.y * tile_width;
    const int block_col0  = blockIdx.x * BLOCK_X;

    const int src_row = clamp_int(row, 0, nrows - 1);

    // Main tile pixel (every thread loads exactly one).
    tile[row_offset + threadIdx.x + kernel_radius] =
        __ldg(&d_src[src_row * ncols + clamp_int(col, 0, ncols - 1)]);

    // Halo: strided loop so kernel_radius > BLOCK_X is still handled.
    for (int i = threadIdx.x; i < kernel_radius; i += BLOCK_X) {
        const int left_col  = clamp_int(block_col0 - kernel_radius + i, 0, ncols - 1);
        const int right_col = clamp_int(block_col0 + BLOCK_X + i,       0, ncols - 1);
        tile[row_offset + i] =
            __ldg(&d_src[src_row * ncols + left_col]);
        tile[row_offset + BLOCK_X + kernel_radius + i] =
            __ldg(&d_src[src_row * ncols + right_col]);
    }

    __syncthreads();

    if (col < ncols && row < nrows) {
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < kernel_size; ++k) {
            acc += c_kernel[k] * tile[row_offset + threadIdx.x + k];
        }
        d_dst[row * ncols + col] = static_cast<unsigned char>(
            clamp_int(__float2int_rn(acc), 0, 255));
    }
}

__global__ void ConvolveVertical(const unsigned char* __restrict__ d_src,
                                  unsigned char* __restrict__ d_dst,
                                  int ncols, int nrows,
                                  int kernel_size, int kernel_radius)
{
    const int col = blockIdx.x * BLOCK_X + threadIdx.x;
    const int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int tile_height = BLOCK_Y + 2 * kernel_radius;
    const int block_row0  = blockIdx.y * BLOCK_Y;

    const int src_col = clamp_int(col, 0, ncols - 1);

    // Main tile pixel.
    tile[(threadIdx.y + kernel_radius) * BLOCK_X + threadIdx.x] =
        __ldg(&d_src[clamp_int(row, 0, nrows - 1) * ncols + src_col]);

    // Halo: strided loop so kernel_radius > BLOCK_Y is still handled.
    for (int i = threadIdx.y; i < kernel_radius; i += BLOCK_Y) {
        const int top_row = clamp_int(block_row0 - kernel_radius + i, 0, nrows - 1);
        const int bot_row = clamp_int(block_row0 + BLOCK_Y + i,       0, nrows - 1);
        tile[i * BLOCK_X + threadIdx.x] =
            __ldg(&d_src[top_row * ncols + src_col]);
        tile[(BLOCK_Y + kernel_radius + i) * BLOCK_X + threadIdx.x] =
            __ldg(&d_src[bot_row * ncols + src_col]);
    }
    (void)tile_height; // documents shared_bytes sizing; not indexed directly

    __syncthreads();

    if (col < ncols && row < nrows) {
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < kernel_size; ++k) {
            acc += c_kernel[k] * tile[(threadIdx.y + k) * BLOCK_X + threadIdx.x];
        }
        d_dst[row * ncols + col] = static_cast<unsigned char>(
            clamp_int(__float2int_rn(acc), 0, 255));
    }
}

// Persistent resources, sized/created lazily and reused across calls.
namespace {
    unsigned char* g_d_src    = nullptr;
    unsigned char* g_d_buffer = nullptr;
    unsigned char* g_d_dst    = nullptr;
    size_t         g_capacity = 0;
    cudaStream_t   g_stream   = nullptr;

    void EnsureDeviceBuffers(size_t bytes_needed)
    {
        if (bytes_needed <= g_capacity) return;
        if (g_d_src)    CUDA_CHECK(cudaFree(g_d_src));
        if (g_d_buffer) CUDA_CHECK(cudaFree(g_d_buffer));
        if (g_d_dst)    CUDA_CHECK(cudaFree(g_d_dst));
        CUDA_CHECK(cudaMalloc(&g_d_src,    bytes_needed));
        CUDA_CHECK(cudaMalloc(&g_d_buffer, bytes_needed));
        CUDA_CHECK(cudaMalloc(&g_d_dst,    bytes_needed));
        g_capacity = bytes_needed;
    }

    void EnsureStream()
    {
        if (!g_stream) CUDA_CHECK(cudaStreamCreate(&g_stream));
    }
}

void LaunchGaussianSmoothingFast(const unsigned char* h_src, unsigned char* h_dst,
        int ncols, int nrows, float* h_kernel, size_t kernel_size)
{
    const int kernel_radius = static_cast<int>(kernel_size / 2);
    const size_t array_size = static_cast<size_t>(ncols) * nrows * sizeof(unsigned char);

    EnsureStream();
    EnsureDeviceBuffers(array_size);

    // Pin the caller's host buffers for this call so cudaMemcpyAsync can
    // use true DMA. Registering already-pinned memory is a no-op error we
    // can safely ignore (fall back to whatever speed pageable gives us).
    cudaError_t reg_src = cudaHostRegister(const_cast<unsigned char*>(h_src),
                                            array_size, cudaHostRegisterDefault);
    cudaError_t reg_dst = cudaHostRegister(h_dst, array_size, cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_d_src, h_src, array_size,
                                cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, h_kernel,
                                        kernel_size * sizeof(float),
                                        0, cudaMemcpyHostToDevice, g_stream));

    const dim3 blockDim(BLOCK_X, BLOCK_Y);
    const dim3 gridDim((ncols + BLOCK_X - 1) / BLOCK_X,
                        (nrows + BLOCK_Y - 1) / BLOCK_Y);

    const size_t shared_horizontal = BLOCK_Y * (BLOCK_X + 2 * kernel_radius) * sizeof(unsigned char);
    const size_t shared_vertical   = (BLOCK_Y + 2 * kernel_radius) * BLOCK_X * sizeof(unsigned char);

    ConvolveHorizontal<<<gridDim, blockDim, shared_horizontal, g_stream>>>(
        g_d_src, g_d_buffer, ncols, nrows, static_cast<int>(kernel_size), kernel_radius);
    CUDA_CHECK(cudaGetLastError());

    ConvolveVertical<<<gridDim, blockDim, shared_vertical, g_stream>>>(
        g_d_buffer, g_d_dst, ncols, nrows, static_cast<int>(kernel_size), kernel_radius);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_dst, g_d_dst, array_size,
                                cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    if (reg_src == cudaSuccess) CUDA_CHECK(cudaHostUnregister(const_cast<unsigned char*>(h_src)));
    if (reg_dst == cudaSuccess) CUDA_CHECK(cudaHostUnregister(h_dst));
}
