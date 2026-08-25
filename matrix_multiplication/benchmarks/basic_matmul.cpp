#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../include/matmul.hpp"
#include "../include/matmul_cuda.hpp"


template <typename T>
void PrintSquareMatrix(const std::vector<T>& mat)
{
    const size_t N = static_cast<size_t>(sqrt(mat.size()));

    std::cout << "[\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << "\t";
        for (size_t j = 0; j < N; ++j) {
            std::cout << mat[i * N + j];
            if (j == N - 1) {
                std::cout << "\n";
            } else {
                std::cout << ", ";
            }
        }
    }
    std::cout << "]\n";
}


void RunBasicMatmul()
{
    constexpr int iterations = 100;

    std::vector<double> mat1 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> dst(mat1.size());

    // ------------------------------------------------------------
    // CPU
    // ------------------------------------------------------------

    auto cpu_start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        SquareMatMul(mat1, mat1, dst);
    }

    auto cpu_end = std::chrono::steady_clock::now();

    const double cpu_total_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start).count();

    std::cout << "\nCPU matrix multiplication:\n";
    std::cout << "\tAverage latency: "
              << cpu_total_ms / iterations << " ms\n";

    PrintSquareMatrix(dst);


    // ------------------------------------------------------------
    // OpenCV
    // ------------------------------------------------------------

    cv::Mat A(3, 3, CV_64F, mat1.data());
    cv::Mat cvDst;

    auto opencv_start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        cvDst = A * A;
    }

    auto opencv_end = std::chrono::steady_clock::now();

    const double opencv_total_ms =
        std::chrono::duration<double, std::milli>(
            opencv_end - opencv_start).count();

    std::cout << "\nOpenCV matrix multiplication:\n";
    std::cout << "\tAverage latency: "
              << opencv_total_ms / iterations << " ms\n";

    std::cout << cvDst << '\n';


    // ------------------------------------------------------------
    // GPU
    // ------------------------------------------------------------

    std::fill(dst.begin(), dst.end(), 0.0f);

    // Warm-up
    LaunchNaiveSqMatMulKernel(mat1, mat1, dst);

    auto gpu_start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        LaunchNaiveSqMatMulKernel(mat1, mat1, dst);
    }

    auto gpu_end = std::chrono::steady_clock::now();

    const double gpu_total_ms =
        std::chrono::duration<double, std::milli>(
            gpu_end - gpu_start).count();

    std::cout << "\nGPU matrix multiplication:\n";
    std::cout << "\tAverage end-to-end latency: "
              << gpu_total_ms / iterations << " ms\n";

    PrintSquareMatrix(dst);
}
