#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include "../include/matmul_cuda.hpp"
#include "../include/benchmarks.hpp"


template <typename T1, typename T2>
void WMMABenchmarkDatatype(
    size_t N,
    const std::string& datatype,
    csvfile& csv,
    size_t iterations = 100)
{
    const size_t elements = N * N;

    std::vector<T1> mat1(elements);
    std::vector<T2> dst(elements);

    // Initialize the matrix.
    for (size_t i = 0; i < elements; ++i) {
        mat1[i] = static_cast<T1>(1.0f);
    }

    // ------------------------------------------------------------
    // GPU
    // ------------------------------------------------------------

    auto matmulGPU = [&]() {
        LaunchWmmaSqMatMulKernel<T1, T2>(mat1, mat1, dst);
    };

    double latencyMicroseconds =
        TimeMatmulLatencyMicroseconds(matmulGPU, iterations);

    csv << N
        << datatype
        << "gpu"
        << latencyMicroseconds
        << endrow;
}


void RunWMMABenchmark()
{
    std::cout << "Tensor cores benchmark\n";

    csvfile csv("./results/TensorCoresBenchmark.csv");

    csv << "size"
        << "datatype"
        << "approach"
        << "latency(us)"
        << endrow;


    constexpr size_t minExponent = 4;   // 2^4  = 16
    constexpr size_t maxExponent = 12;  // 2^12 = 4096

    constexpr size_t iterations = 100;


    for (size_t exponent = minExponent;
         exponent <= maxExponent;
         ++exponent)
    {
        const size_t N = size_t{1} << exponent;

        std::cout << "\nMatrix size: "
                  << N << "x" << N << '\n';


        // --------------------------------------------------------
        // Floating point
        // --------------------------------------------------------

        std::cout << "  fp16...\n";

        WMMABenchmarkDatatype<__half, float>(
            N, "half", csv, iterations);

        std::cout << "  bf16...\n";

        WMMABenchmarkDatatype<__nv_bfloat16, float>(
            N, "bf16", csv, iterations);
    }

    std::cout << "\nBenchmark complete.\n";
}
