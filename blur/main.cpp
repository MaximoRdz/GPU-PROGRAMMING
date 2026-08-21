#include <chrono>
#include <iostream>
#include <cstddef>
#include <vector>

#include <opencv2/opencv.hpp>

#include "blur_cpu.hpp"
#include "blur_cuda.hpp"
#include "blur_cuda_fast.hpp"
#include "example.hpp"

#include "opencv2/highgui.hpp"
#include "utils.hpp"


int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return 1;
    }
    
    cv::Mat raw = cv::imread(argv[1]);
    if (raw.empty()){
        std::cerr << "ERROR: Could not open image: " << argv[1] << std::endl;
        return 1;
    }
    cv::cvtColor(raw, raw, cv::COLOR_BGR2GRAY);


    const size_t iterations = 100;
    const double sigma = 10.0;
    const size_t kernel_size = KernelSizeForSigma(sigma);
    std::vector<double> kernel(kernel_size);
    ComputeGaussianKernel1D(kernel.data(), kernel_size, sigma);

    std::cout << "Experiment Sigma: " << sigma << " kernel-size: "
        << kernel_size << std::endl;

    cv::Mat blurred(raw.rows, raw.cols, CV_8UC1);

    std::chrono::high_resolution_clock::time_point start, end;
    double avg_duration_microseconds;
    /* /// CPU Custom Implementation */
    /* start = TickNow(); */
    /* for (size_t i = 0; i < iterations; ++i){ */
    /*     GaussianFilterCPU(raw.data, blurred.data, raw.rows, raw.cols, */
    /*             kernel.data(), kernel_size); */
    /* } */
    /* end = TickNow(); */

    /* avg_duration_microseconds = DurationMicroseconds(start, end) / */ 
    /*     static_cast<double>(iterations); */

    /* std::cout << "Avg. Duration CPU approach: " << avg_duration_microseconds */ 
    /*     << " microseconds / iteration"<< std::endl; */
    
    /// CPU OpenCV Implementation
    
    // warmup
    cv::GaussianBlur(raw, blurred, cv::Size(kernel_size, kernel_size), 0, 0);

    start = TickNow();
    for (size_t i = 0; i < iterations; ++i){
        cv::GaussianBlur(raw, blurred, cv::Size(kernel_size, kernel_size), 0, 0);
    }
    end = TickNow();

    avg_duration_microseconds = DurationMicroseconds(start, end) / 
        static_cast<double>(iterations);

    std::cout << "Avg. Duration CPU approach: " << avg_duration_microseconds 
        << " microseconds / iteration"<< std::endl;

    /// GPU Implementation
    std::vector<float> kernel_f(kernel.begin(), kernel.end());

    // warmup
    start = TickNow();
    LaunchGaussianSmoothing(raw.data, blurred.data, raw.cols, raw.rows, kernel_f.data(), kernel_size);
    end = TickNow();
    avg_duration_microseconds = DurationMicroseconds(start, end);
    std::cout << "\tSingle warmup iteration in GPU: " << avg_duration_microseconds 
        << " microseconds" << std::endl;


    start = TickNow();
    for (size_t i = 0; i < iterations; ++i){
        LaunchGaussianSmoothing(raw.data, blurred.data, raw.cols, raw.rows, kernel_f.data(), kernel_size);
    }
    end = TickNow();

    avg_duration_microseconds = DurationMicroseconds(start, end) / 
        static_cast<double>(iterations);

    std::cout << "Avg. Duration GPU approach: " << avg_duration_microseconds 
        << " microseconds / iteration"<< std::endl;

    /// GPU Implementation All tricks

    // warmup
    start = TickNow();
    LaunchGaussianSmoothingFast(raw.data, blurred.data, raw.cols, raw.rows, kernel_f.data(), kernel_size);
    end = TickNow();

    avg_duration_microseconds = DurationMicroseconds(start, end);

    std::cout << "\tSingle warmup iteration in GPU: " << avg_duration_microseconds 
        << " microseconds" << std::endl;


    start = TickNow();
    for (size_t i = 0; i < iterations; ++i){
        LaunchGaussianSmoothingFast(raw.data, blurred.data, raw.cols, raw.rows, kernel_f.data(), kernel_size);
    }
    end = TickNow();

    avg_duration_microseconds = DurationMicroseconds(start, end) / 
        static_cast<double>(iterations);

    std::cout << "Avg. Duration GPU approach: " << avg_duration_microseconds 
        << " microseconds / iteration"<< std::endl;
    /* DisplayExample(raw); */
    

    double scale = 1024.0 / std::max(blurred.rows, blurred.cols);
    cv::resize(blurred, blurred, cv::Size(), scale, scale);
    cv::imshow("GPU result", blurred);
    cv::waitKey(0);

    return 0;
}

