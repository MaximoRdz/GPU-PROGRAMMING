#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include "../include/matmul.hpp"
#include "../include/matmul_cuda.hpp"
#include "../include/benchmarks.hpp"


template <typename T>
void TilingBenchmarkDatatype(
    size_t N,
    const std::string& datatype,
    csvfile& csv,
    size_t iterations = 100)
{
    const size_t elements = N * N;

    std::vector<T> mat1(elements);
    std::vector<T> dst(elements);

    // Initialize the matrix.
    for (size_t i = 0; i < elements; ++i) {
        mat1[i] = static_cast<T>(1.0f);
    }

    // ------------------------------------------------------------
    // GPU
    // ------------------------------------------------------------

    auto matmulGPU = [&]() {
        LaunchOptimizedSqMatMulKernel<T>(mat1, mat1, dst);
    };

    double latencyMicroseconds =
        TimeMatmulLatencyMicroseconds(matmulGPU, iterations);

    csv << N
        << datatype
        << "gpu"
        << latencyMicroseconds
        << endrow;
}


void RunTilingBenchmark()
{
    std::cout << "Math-bound tilnig benchmark\n";

    csvfile csv("./results/TilingBenchmark.csv");

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
        // Integer types
        // --------------------------------------------------------

        std::cout << "  int16...\n";

        TilingBenchmarkDatatype<std::int16_t>(
            N, "int16", csv, iterations);

        // --------------------------------------------------------
        // Floating point
        // --------------------------------------------------------

        std::cout << "  fp16...\n";

        TilingBenchmarkDatatype<__half>(
            N, "half", csv, iterations);

        std::cout << "  bf16...\n";

        TilingBenchmarkDatatype<__nv_bfloat16>(
            N, "bf16", csv, iterations);
    }

    std::cout << "\nBenchmark complete.\n";
}
