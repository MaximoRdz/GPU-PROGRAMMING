#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../include/matmul.hpp"
#include "../include/matmul_cuda.hpp"
#include "../include/benchmarks.hpp"


/*
 * Resutls
 *
 * matrix sizes: 16, 32, 64, 128, ..., 2048
 *    e.g. 2^4 to 2^11
 *
 * output csv with:
 * matrix size | data-type | end2end latency
 *
 * data-types:
 * - int8, int16, int32, int64
 * - bf16, f16, float (32), double (64)
 *
 * */
template <typename T>
void BenchmarkDatatype(
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
        mat1[i] = static_cast<T>(1);
    }

    // ------------------------------------------------------------
    // CPU
    // ------------------------------------------------------------

    auto matmulCPU = [&]() {
        SquareMatMul(mat1, mat1, dst);
    };

    double latencyMicroseconds =
        TimeMatmulLatencyMicroseconds(matmulCPU, iterations);

    csv << N
        << datatype
        << "cpu"
        << latencyMicroseconds
        << endrow;


    // ------------------------------------------------------------
    // GPU
    // ------------------------------------------------------------

    auto matmulGPU = [&]() {
        LaunchNaiveSqMatMulKernel(mat1, mat1, dst);
    };

    latencyMicroseconds =
        TimeMatmulLatencyMicroseconds(matmulGPU, iterations);

    csv << N
        << datatype
        << "gpu"
        << latencyMicroseconds
        << endrow;
}


void RunDatatypeBenchmark()
{
    std::cout << "Datatype benchmark\n";

    csvfile csv("./results/DatatypeBenchmark.csv");

    csv << "size"
        << "datatype"
        << "approach"
        << "latency(us)"
        << endrow;


    constexpr size_t minExponent = 4;   // 2^4  = 16
    constexpr size_t maxExponent = 10;  // 2^10 = 1024

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

        std::cout << "  int8...\n";

        BenchmarkDatatype<std::int8_t>(
            N, "int8", csv, iterations);

        std::cout << "  int16...\n";

        BenchmarkDatatype<std::int16_t>(
            N, "int16", csv, iterations);

        std::cout << "  int32...\n";

        BenchmarkDatatype<std::int32_t>(
            N, "int32", csv, iterations);

        std::cout << "  int64...\n";

        BenchmarkDatatype<std::int64_t>(
            N, "int64", csv, iterations);


        // --------------------------------------------------------
        // Floating point
        // --------------------------------------------------------

        std::cout << "  float...\n";

        BenchmarkDatatype<float>(
            N, "float32", csv, iterations);

        std::cout << "  double...\n";

        BenchmarkDatatype<double>(
            N, "float64", csv, iterations);
    }

    std::cout << "\nBenchmark complete.\n";
}
