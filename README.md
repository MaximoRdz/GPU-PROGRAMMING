# CUDA C++ Programming Guide
`CUDA`: Introduced in November 2006, CUDA is a parallel computing platform and programming model for accelerating compute-intensive applications using NVIDIA GPUs.
## Introduction
While the CPU is designed to excel at executing sequences of operations (threads) as fast as possible and can execute a few tens of them simultaneously (from 4–8 in basic consumer CPUs to 32, 64, and up to 128+ in server-grade processors), a GPU is designed to excel at executing *thousands* of threads in parallel. This provides a good trade-off between slower single-thread performance and the vast number of threads a GPU can handle.

In a nutshell, GPUs are specialized for highly parallel computations. Hardware-wise, many more transistors are dedicated to data processing rather than data caching and flow control. Nicely illustrated in the following figure:

![CPU-GPU architecture](assets/cpu-gpu-arch.png)

> CUDA programming: Make parallel software scale transparently with a low learning curve for programmers coming from languages such as C.

### Fundamentals

1. **Hierarchy of thread groups**
2. **Shared memory**
3. **Barrier synchronization**

These fundamental abstractions provide fine-grained data parallelism and thread parallelism, nested with coarse-grained data parallelism and task parallelism. They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.

> **Note:** A GPU is built around an array of *Streaming Multiprocessors* (SMs). A multithreaded program is partitioned into blocks of threads that can execute independently. A GPU with more multiprocessors can generally execute more blocks concurrently, although performance does not scale automatically because it also depends on memory bandwidth, instruction mix, synchronization, and other resources.

## Programming Model

### Kernels

The idea is to define C++ functions, called kernels, that, when called, are executed by many CUDA threads in parallel.

- Kernel: `__global__` declaration specifier and the number of CUDA threads that execute the kernel for a given kernel call (`<<<...>>>`).
- Each thread is given a unique thread ID, accessible within the kernel through built-in variables.

```cu
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

### Warps

A **warp** is a group of **32 CUDA threads** that the GPU schedules and executes together using the SIMT (Single Instruction, Multiple Threads) model.

For example, a block with 256 threads contains:

```text
256 threads / 32 threads per warp = 8 warps
```

Threads in the same warp normally execute the same instruction at the same time, although each thread has its own registers and data. If threads in a warp take different branches, **warp divergence** can reduce performance because the different paths may need to be executed separately.

> **Note:** A warp is smaller than a thread block. A block can contain multiple warps; for example, the DGX Spark's maximum 1024-thread block contains 32 warps.


### Thread Hierarchy

`threadIdx` is a 3-component vector, so threads can be identified in a 1D array, a 2D grid, or a 3D lattice. This provides a convenient way to map computations onto elements of a vector, matrix, or volume.

- Relation between the index of a given thread and its assigned linear thread ID:
  - 1D: `x`
  - 2D block of size `(Dx, Dy)`: `x + y * Dx`
  - 3D block: `x + y * Dx + z * Dx * Dy`

```cu
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N x N threads.
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

Of course, there is a limit on the number of threads per block, since all threads of a block are scheduled on the same Streaming Multiprocessor (SM) and share limited SM resources. On current CUDA GPUs, the maximum is **1024 threads per block**.

In my specific current case:

> **NVIDIA DGX Spark:** 48 SMs and 1024 threads per block.

The DGX Spark uses the GB10 Grace Blackwell GPU, with 6,144 CUDA cores. The 48-SM figure and the 1024-thread block limit are hardware/architecture properties; the number of SMs does not determine the maximum threads per block.

Each block within the grid can be identified by a one-, two-, or three-dimensional unique index, accessible through `blockIdx`. The dimensions of the thread block are accessible through `blockDim`. The previous example can be extended to handle multiple blocks:

```cu
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main()
{
    ...
    // Kernel invocation with multiple blocks.
    dim3 threadsPerBlock(16, 16); // 256 threads in a block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

For example, if `N = 128`, `numBlocks` is `(8, 8)`: 64 blocks, each with 256 threads, for **16,384 total threads** in the grid. This is not too much for the DGX Spark: the grid is a logical workload, and blocks are scheduled onto the available SMs as resources become available. They do not all need to execute simultaneously.

- Threads within a block can share data through shared memory and synchronize with `__syncthreads()`.

### Thread Block Clusters

Until now, threads were organized in 1D, 2D, or 3D blocks, and blocks were organized into a grid. CUDA also provides another optional level of hierarchy: **thread block clusters**, which group thread blocks in 1D, 2D, or 3D.

On compute capability 9.0 and later, the portable maximum cluster size is **8 thread blocks**. Larger architecture-specific limits may exist and should be queried when needed.

So the hierarchy can be summarized as:

**Grid → Thread Block Cluster → Thread Block → Thread → Warp**

### Memory Hierarchy
Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. each
thread block in a thread block clustre can perform read/write/atomic operations on each other's shared memory. **all threads have access to the same global memory**

**read only** spaces: two additional memory spaces accessible by all threads: *constant* and *texture* moemory spaces. They are optimized for different memory usages.

### Heterogeneous Programming
![het-programming](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/_images/heterogeneous-programming.png)

# References

- https://docs.nvidia.com/cuda/cuda-programming-guide/
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
