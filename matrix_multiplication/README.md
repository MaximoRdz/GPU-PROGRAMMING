# Matrix Multiplication
C++ and Cuda implementations and optimization experiments.

main proble of matrix Multiplication via naive implementation:
once each thread is assigned a target dst (row, col) it has to perform an accumulation over the
multiplication of matrices `a` and `b`row and col elements (as given by the eqn). This accumultaiton in the gpu basically retrives from global memory `d_a[]` and `d_b[]`elements (in float precision 4 bytes) and writes in the result element `d_dst[]` another number. so two operations `*` and `+` for 8 bytes retrieved and another 4 bytes wrtten after. Really inefficient and bandwith bound for small size matrices.

## Naive implementation
### CPU vs GPU
Naive implementation results averaged across 100 iterations, measuring end-to-end latency. As show in the first image below, we distinguish the two expected regimes:
(1) memory bound and (2) compute bound. (1) For matrix sizes smaller than 128$\times$128 the gpu added overhead (memory allocation via cudaMalloc and memory retrival in parallel by the 
threads when computing the multiplication) dominates and even having thounsand of threads in parallel CPU single threaded implementaiton is still faster.
<img src="./results/DatatypeBenchmark.png" width="850">

### GPU only
The following figures show how the end-to-end latency is fixed and stable for matrix sizes up to 256$\times$256 and for all data types. What does this mean? The overhead of preparing the gpu and launching the kernel dominates the execution time, thus having more elements in this regime does not make a difference. It is only for matrices larger than 512$\times$512 where we start to see the exponential growth of the actual arithmetic work (due to having more elements).

<img src="./results/DatatypeBenchmark_gpu.png" width="850">

Fastest datatypes? `int16` in the compute bound regime and `int8` in the memory-overhead regime (make sense). IN the compute-bound regime, the most interesting, `int16` is followed by `int8`, `float32` and `int32`.
| Size    |     16 |     32 |     64 |    128 |    256 |     512 |    1024 |
| ------- | -----: | -----: | -----: | -----: | -----: | ------: | ------: |
| float32 | 173.97 | 172.19 | 175.61 | 195.93 | 267.76 |  960.47 | 5245.05 |
| float64 | 168.87 | 172.26 | 180.37 | 202.82 | 352.94 | 1734.14 | 10638.4 |
| int16   | **168.59** |    171 | 180.14 | 188.72 | **221.55** |  **546.78** | **3225.88** |
| int32   | 172.55 | 171.31 | 176.08 | 195.88 | 268.35 |  963.03 | 5234.65 |
| int64   | 169.61 | 172.52 | 179.75 | 201.63 | 351.36 | 1730.39 | 10676.1 |
| int8    | 170.25 | **170.51** | **175.26** | **185.01** | 279.51 |  845.56 | 5077.68 |

### GPU only i16, fp16 and bf16
Currently, three of the most important datatypes in machine learning. `bf16` introduces some overhead over the other types. This has to do with the number fundamental definition:
- fp16: one sign bit, five exponent bits and ten mantissa bits (represents numbers with roughly three to four decimal precision)
- bf16: one sign bit, eight exponent bits and seven mantissa bits (represents a wider range of numbers but decimal precision gets
reduced to two to three digits)

It is clear that `bf16` sacrifices precision for broader exponent range.

In this experiment we see similar performance, but architecture favours fp16


<p float="left">
    <img src="./results/GPUi16fp16bf16Benchmark.png" width="49%">
    <img src="./results/GPUi16fp16bf16Benchmark_linear.png" width="49%">
</p>

## Matrix Multiplication Optimization
Accelerate the data traffic without affecting the number of math operations.

### Math-bound vs Memory-bound Operations
An operation performed on a piece of hardware can be either math- or memory-bound.
* math bandwith: the rate at which math unit operation can be conducted by the processor `operations/second` (OPS) if we are working with float datatuype then the most common name is FLOPS (my nvidia dgx shows off $\approx$100 TFLOPS, i.e. $10^{12}$ FLOPS)
* memory bandwidth: rate at which data can be read from or stored into a semiconductor memory by a processor `bytes / second`, again the nvidia dgx 273 GBytes/s 
* data reuse: cache, smaller faster memory closer to the processor core, data reuse is just the idea of copy the data you are going to be using a lot to cache.

Matrix multiplication requires $2*N^3$ math operations (addition and multiplication) in the naive approach. If we are using a data precision of `b` bits then the amoutn of data read is $2b*N^3$ let's say in the cache memory we can store a whole N$\times$N matrix a one array of $N$ elements, the we can reduce the amount of data reads to $N\times$N + N$\times$N (moving first marix to cache and the reading and moving array by array next matrix to cache) and for`b` bits $2bN^2$. in terms of
writing to memory, just write the otuput matrix $bN^2$, so `data transfer`-wise there wil be $3bN^2$.

#### definitions
time it takes in terms of operations
$$t_{math} = \frac{N_{op}}{BW_{math}}$$

time it takes in temrs of memory operaitons
$$t_{mem} = \frac{N_{byte}}{BW_{mem}}$$

hence we can see which dominates the total time:
- math-bound
$$\frac{N_{op}}{N_{byte}} \gt \frac{BW_{math}}{BW_{mem}}$$

- memory-bound
$$\frac{N_{op}}{N_{byte}} \lt \frac{BW_{math}}{BW_{mem}}$$

an using the right terminology $\frac{N_{op}}{N_{byte}}$ is referred to as the **arithmetic intensity**

### Optimizing
$N_{op}$ is usually a constant unless you know better algorithms, but we can try to minimize $N_{bytes}$ as much as possible by reusing data. If an operation is memory bound then the computer-system is under-utilized.
- matrix multiplication case: for this case the arithmetic intensity becomes
$$\frac{N_{op}}{N_{byte}} = \frac{2N^3}{2bN^3 / 8} = \frac{8}{b} \; \text{OP/byte}$$
and in the case of data reuse (as described above)
$$\frac{N_{op}}{N_{byte}} = \frac{2N^3}{3bN^2 / 8} = \frac{16N}{3b} \; \text{OP/byte}$$

for my nvidia DGX spark: $\frac{BW_{math}}{BW_{mem}} = \frac{10^{12} FLOPS}{273 \cdot 10^9 B/s} = 3.663$ OP/byte arithmetic intensity, under this threshold the operation is considered memory bound. As this specifications of the dgx spark are for fp16 the $\frac{N_{op}}{N_{byte}} = \frac{16N}{3b} = \frac{N}{3} \; \text{OP/byte}$

so if $N \lt 3 \cdot 3.663 \approx 10$ the multiplicatoin operation is memory bound


## Tensor Cores and Matrix Multiplication
- https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/
- https://leimao.github.io/blog/CUDA-Matrix-Multiplication/
- https://docs.nvidia.com/cuda/archive/11.6.1/cuda-c-best-practices-guide/index.html

NVIDIA tensor cores: dedicated (hardware) accelerators for general matrix multiplication (GEMM). NVIDIA tensor cores are specialized 
in performing the GEMM operations in mixed precision, i.e., GEMM inputs are in lower precision whereas GEMM outputs are in high precision.

Currently, I'm working on a NVIDIA DGX Spark, grace blackwell architecture, which accounts for 48 SMs and each has 4 tensor cores, thus 192 tensor cores in total. Another example is the A100 with 108 SMs, also 4 TC / Sm and hence a total of 432 tensor cores.

### Programming Tensor Cores
TC are fully prgrammable at the warp level and the API can be access via `mma.h`(matrix multiplication and accumulation) and under the `nvcuda::wmma` namespace.
We can program operations of the type:
$$D = AB + C$$
at the warp level (32 threads). Given a large input matrix, the full MMA operation can be divided into multiple small GEMMs by dividing the matrices into smaller 
matrices and then collect the final result as follows:


## Misc.
- **lambda function in c++**
```c++
[captures](parameters) {
    body
}
```
`[&]` allows the lambda function to access variables from the surrounding scope
and use them, only by reference!
# TODO
- [x] performance for different data types and floating point precisions.
- [ ] optimize approach: math-bound vs compute-bound
- [ ] optimize with share memory
- [ ] optmize with tensor cores
- [ ] compare against cuDNN or any oficial GEMM implementaiton
- batched matrix multiplication

