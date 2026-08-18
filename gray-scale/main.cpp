#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "grayscale_cuda.h"

void myGrayImage(const cv::Mat& image, cv::Mat& output_image)
{
    // RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
    for (int r=0; r < image.rows; ++r){
        for (int c=0; c < image.cols; ++c){
            cv::Vec3b pixel_value = image.at<cv::Vec3b>(r, c);
            uchar B = pixel_value[0];
            uchar G = pixel_value[1];
            uchar R = pixel_value[2];

            output_image.at<uchar>(r, c) = 0.299f * R + 0.587 * G + 0.114 * B;
        }
    }
}

void myGrayImageOpt(const cv::Mat& image, cv::Mat& output_image)
{
    // RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
    for (int r=0; r < image.rows; ++r){
        const uchar* src_row = image.ptr<uchar>(r);
        uchar* dst_row = output_image.ptr<uchar>(r);
        for (int c=0; c < image.cols; ++c){
            uchar B = src_row[c * 3 + 0];
            uchar G = src_row[c * 3 + 1];
            uchar R = src_row[c * 3 + 2];

            dst_row[c] = (uchar)(0.299f * R + 0.587 * G + 0.114 * B);
        }
    }
}

double maxDiff(const cv::Mat& A, const cv::Mat& B)
{
    cv::Mat diff;
    cv::absdiff(A, B, diff);
    
    double max_diff;
    cv::minMaxLoc(diff, nullptr, &max_diff);

    return max_diff;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    cv::Mat image = cv::imread(filename);

    if (image.empty()){
        std::cerr << "Error: Could not load image: "
                  << filename << std::endl;
        return 1;
    }

    cv::Mat resized;
    double scale = 1024.0f / std::max(image.cols, image.rows);
    cv::resize(image, resized, cv::Size(), scale, scale);

    cv::Mat gray_image_cv(resized.rows, resized.cols, CV_8UC1);
    cv::Mat gray_image_my(resized.rows, resized.cols, CV_8UC1);
    cv::Mat gray_image_gpu(resized.rows, resized.cols, CV_8UC1);

    const size_t iterations = 100;
    cv::cvtColor(resized, gray_image_cv, cv::COLOR_BGR2GRAY);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        cv::cvtColor(resized, gray_image_cv, cv::COLOR_BGR2GRAY);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_execution_time = duration.count() / (double)iterations;

    std::cout << "OpenCV Average time per call: " << avg_execution_time << " microseconds\n";

    
    myGrayImage(resized, gray_image_my);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        myGrayImage(resized, gray_image_my);
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_execution_time = duration.count() / (double)iterations;

    std::cout << "myGrayImage Average time per call: " << avg_execution_time << " microseconds\n";

    myGrayImageOpt(resized, gray_image_my);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        myGrayImageOpt(resized, gray_image_my);
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_execution_time = duration.count() / (double)iterations;

    std::cout << "myGrayImageOpt Average time per call: " << avg_execution_time << " microseconds\n";

    std::cout << "step in the matrix: " << gray_image_gpu.step << std::endl;
    launchGrayScaleConversion(resized.data, gray_image_gpu.data, resized.cols, resized.rows,
                               resized.channels(), resized.step, gray_image_gpu.step,
                               nullptr, nullptr);

    double totalKernelMs = 0.0, totalFullMs = 0.0;
    for (size_t i = 0; i < iterations; ++i) {
        float kernelMs = 0.0f, fullMs = 0.0f;
        launchGrayScaleConversion(resized.data, gray_image_gpu.data, resized.cols, resized.rows,
                                   resized.channels(), resized.step, gray_image_gpu.step,
                                   &kernelMs, &fullMs);
        totalKernelMs += kernelMs;
        totalFullMs += fullMs;
    }
    std::cout << "CUDA kernel-only Average time per call: "
              << (totalKernelMs / iterations) * 1000.0 << " microseconds\n";
    std::cout << "CUDA total (incl. transfers) Average time per call: "
              << (totalFullMs / iterations) * 1000.0 << " microseconds\n";

    // compare matrices
    std::cout << "matrices [cv, my] max abs diff " << maxDiff(gray_image_cv, gray_image_my) << std::endl;
    std::cout << "matrices [cv, gpu] max abs diff " << maxDiff(gray_image_cv, gray_image_gpu) << std::endl;
    std::cout << "matrices [my, gpu] max abs diff " << maxDiff(gray_image_my, gray_image_gpu) << std::endl;
    // Finally some viz
    cv::imshow("Image Viewer", gray_image_gpu);

    cv::waitKey(0);

    return 0;
}
