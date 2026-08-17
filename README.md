# CUDA C++ Programming Guide
`CUDA`: Introduced in November 2006, CUDA is a parallel computing platform and programming model allowing to accelerate
compute-intensive applications using NVIDIA's GPUs.

## Introduction
While the CPU is designed to excel  at execution of sequence of operations (threads), as fast as
possible and can execute a few tens of them simultaneously (from 4-8 in basic consumer cpus to 
32, 64 and up to 128+ in server grade processors). A GPU is designed to excel at executing *thousands*
of them in parallel, providing a good trade off between their slower single thread performance and the
vast amount of them it can handle.

In a nutshell, GPUs are speciallized for highly parallel computations, hardware-wise much more transistors 
are dedicated for data processing rather than data caching and flow control. Nicely illustrated in the
following figure
![cpu-gpu-arch](assets/cpu-gpu-arch.png)

> CUDA programming: Make parallel software scale transparently with a low learning curve for
> programmers coming from languages such as C

* `fundamentals`:
    1. hierarchy of thread groups
    1. shared memories
    1. barrier synchronization
* These fundamental abstractions provide fine-grained data parallelism and thread parallelism, nested with
coarse-grained data parallelism and task parallelism. Guiding the programmer to partition the problem into coarse
sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer
pieces that can be solved cooperatively in parallel by all threads withing the block.
* `Note`: GPU is built around an array of *Streaming Multiprocessors* (SMs). A multithreaded program is partitioned into blocks
of threads that execute indepentdenly to each other, GPU with more multiprocessors will automatically execute the program faster.

## Programming Model
### Kernels
Idea: define C++ functions, called kernels, that, when called are executed N times in parallel by N different CUDA threads.
- Kernel: `__global__` declaration specifier and the number of cuda threads that execute the kernel for a given kernel call (<<<...>>>).
Each thread is given a unique thread ID (accessible withing the kernel through built-in variables).
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

### Thread Hierachy
`threadIdx` is a 3 component vector, hence threads can be identified in an array (1D), in a grid (2D) or lattice (3D), providing a
really convinient approach to invoke computations across elements of vector, matrix or volumes.
- Relation between index of a given thread and its assignated thread ID: 1D array -> the same, 2D block of size (Dx, Dy) `y*Dx + x`
and in a 3D block `x + y*Dx + z*Dx*Dy`

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
    // kernel invocatoin with one block of NxN threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
Of course, there is a limit on number fo threads per block, since all thread of a block are expected to reside on the same 
Streaming Multiprocessor (SM) core and must share limited memory resources. On current GPUs up 1024 threads. In my specific crrent case:

> Nvidia DGX spark 48 SMs and 1024 threads per block. 

Each block within the grid can be identified by a one-dimensional, two-dimension or three-dimensional unique index, and it is accessible
through `blockIdx` variable, the dmiension of the thread block is accessibl via `blockDim`, the previous example can be extended to handle
multiple blokcs:


```cu
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j <N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main()
{
    ...
    // kernel invocatoin with one block of NxN threads
    dim3 threadsPerBlock(16, 16); // 256 threads in a block
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y); // e.g. N=128 then (8, 8) meaning 64 blocks each with 256 threads 16384 total threads?? is this too much for my DGX?
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
* *Threads within a block can share data: `__syncthreads()`.

- thread block clusters: Until now we could have threads organized in 1-2-or-3 Dimensions, but also organize threads by groups of threads in 1-2-or-3 dimensions that could cooperated, but now
we wnat to introduce one more level of hierarchy, thread block clusters again in 1-2-or-3 dimensions. currently a maximum of 8 thread blocks in a cluster. Summerizing you could potentially have
3 * 3 * 3 = 27 dimensions of parallelization.




# References
- https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html
# GPU-PROGRAMMING
