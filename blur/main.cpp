#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "blur_cpu.hpp"
#include "blur_cuda.hpp"
#include "blur_cuda_fast.hpp"
#include "example.hpp"

#include "opencv2/highgui.hpp"
#include "utils.hpp"

#define MAX_KERNEL_SIZE 64  // must match __constant__ c_kernel[] size in both .cu files

struct BenchmarkResult {
    double sigma;
    size_t kernel_size;
    double cpu_custom_us;
    double opencv_us;
    double gpu_tiled_us;
    double gpu_fast_us;
};

static void WriteCsv(const std::string& path, const std::vector<BenchmarkResult>& results)
{
    std::ofstream out(path);
    out << "sigma,kernel_size,cpu_custom_us,opencv_us,gpu_tiled_us,gpu_fast_us\n";
    for (const auto& r : results) {
        out << r.sigma << "," << r.kernel_size << ","
            << r.cpu_custom_us << "," << r.opencv_us << ","
            << r.gpu_tiled_us << "," << r.gpu_fast_us << "\n";
    }
    std::cout << "Wrote " << path << std::endl;
}

static void PlotWithGnuplot(const std::string& csv_path, const std::string& output_png)
{
    FILE* gp = popen("gnuplot", "w");
    if (!gp) {
        std::cerr << "Could not launch gnuplot (install it: sudo apt install gnuplot)\n";
        return;
    }
    fprintf(gp, "set datafile separator ','\n");
    fprintf(gp, "set title 'Gaussian Blur: Avg Duration vs Sigma'\n");
    fprintf(gp, "set xlabel 'Sigma'\n");
    fprintf(gp, "set ylabel 'Avg Duration (microseconds, log scale)'\n");
    fprintf(gp, "set logscale y\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key outside right\n");
    fprintf(gp, "set terminal pngcairo size 1100,750 font 'sans,11'\n");
    fprintf(gp, "set output '%s'\n", output_png.c_str());
    fprintf(gp,
        "plot '%s' using 1:3 with linespoints lw 2 pt 7 title 'CPU custom', "
        "''    using 1:4 with linespoints lw 2 pt 7 title 'OpenCV CPU', "
        "''    using 1:5 with linespoints lw 2 pt 7 title 'GPU tiled (mine)', "
        "''    using 1:6 with linespoints lw 2 pt 7 title 'GPU fast'\n",
        csv_path.c_str());
    pclose(gp);
    std::cout << "Plot saved to " << output_png << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return 1;
    }

    cv::Mat raw = cv::imread(argv[1]);
    if (raw.empty()) {
        std::cerr << "ERROR: Could not open image: " << argv[1] << std::endl;
        return 1;
    }
    cv::cvtColor(raw, raw, cv::COLOR_BGR2GRAY);

    const size_t iterations = 30; // reduced from 100 - sweep runs every approach per sigma
    const std::vector<double> sigmas = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    cv::Mat blurred(raw.rows, raw.cols, CV_8UC1);
    std::vector<BenchmarkResult> results;

    std::chrono::high_resolution_clock::time_point start, end;

    for (double sigma : sigmas) {
        const size_t kernel_size = KernelSizeForSigma(sigma);

        if (kernel_size > MAX_KERNEL_SIZE) {
            std::cerr << "Skipping sigma=" << sigma << " (kernel_size=" << kernel_size
                       << " exceeds MAX_KERNEL_SIZE=" << MAX_KERNEL_SIZE
                       << " constant-memory limit)\n";
            continue;
        }

        std::vector<double> kernel(kernel_size);
        ComputeGaussianKernel1D(kernel.data(), kernel_size, sigma);
        std::vector<float> kernel_f(kernel.begin(), kernel.end());

        std::cout << "Experiment Sigma: " << sigma << " kernel-size: " << kernel_size << std::endl;

        BenchmarkResult r;
        r.sigma = sigma;
        r.kernel_size = kernel_size;

        /// CPU Custom Implementation
        GaussianFilterCPU(raw.data, blurred.data, raw.rows, raw.cols,
                           kernel.data(), kernel_size); // warmup, untimed
        start = TickNow();
        for (size_t i = 0; i < iterations; ++i) {
            GaussianFilterCPU(raw.data, blurred.data, raw.rows, raw.cols,
                               kernel.data(), kernel_size);
        }
        end = TickNow();
        r.cpu_custom_us = DurationMicroseconds(start, end) / static_cast<double>(iterations);
        std::cout << "  CPU custom:  " << r.cpu_custom_us << " us/iter\n";

        /// CPU OpenCV Implementation
        cv::GaussianBlur(raw, blurred, cv::Size(kernel_size, kernel_size), 0, 0); // warmup
        start = TickNow();
        for (size_t i = 0; i < iterations; ++i) {
            cv::GaussianBlur(raw, blurred, cv::Size(kernel_size, kernel_size), 0, 0);
        }
        end = TickNow();
        r.opencv_us = DurationMicroseconds(start, end) / static_cast<double>(iterations);
        std::cout << "  OpenCV CPU:  " << r.opencv_us << " us/iter\n";

        /// GPU Implementation (tiled, mine)
        start = TickNow();
        LaunchGaussianSmoothing(raw.data, blurred.data, raw.cols, raw.rows,
                                 kernel_f.data(), kernel_size); // warmup, untimed, NOT divided
        end = TickNow();
        std::cout << "  GPU tiled warmup: " << DurationMicroseconds(start, end) << " us\n";

        start = TickNow();
        for (size_t i = 0; i < iterations; ++i) {
            LaunchGaussianSmoothing(raw.data, blurred.data, raw.cols, raw.rows,
                                     kernel_f.data(), kernel_size);
        }
        end = TickNow();
        r.gpu_tiled_us = DurationMicroseconds(start, end) / static_cast<double>(iterations);
        std::cout << "  GPU tiled:   " << r.gpu_tiled_us << " us/iter\n";

        /// GPU Implementation (fast, all tricks)
        start = TickNow();
        LaunchGaussianSmoothingFast(raw.data, blurred.data, raw.cols, raw.rows,
                                     kernel_f.data(), kernel_size); // warmup, untimed
        end = TickNow();
        std::cout << "  GPU fast warmup:  " << DurationMicroseconds(start, end) << " us\n";

        start = TickNow();
        for (size_t i = 0; i < iterations; ++i) {
            LaunchGaussianSmoothingFast(raw.data, blurred.data, raw.cols, raw.rows,
                                         kernel_f.data(), kernel_size);
        }
        end = TickNow();
        r.gpu_fast_us = DurationMicroseconds(start, end) / static_cast<double>(iterations);
        std::cout << "  GPU fast:    " << r.gpu_fast_us << " us/iter\n\n";

        results.push_back(r);
    }

    const std::string csv_path = "benchmark_results.csv";
    WriteCsv(csv_path, results);

    double scale = 1024.0 / std::max(blurred.rows, blurred.cols);
    cv::resize(blurred, blurred, cv::Size(), scale, scale);
    cv::imshow("GPU result", blurred);
    cv::waitKey(0);

    return 0;
}
