#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "blur_cuda.h"

namespace {

enum class Axis { kHorizontal, kVertical };

void ComputeGaussianKernel1D(double* kernel, size_t kernel_size, double sigma)
{
    assert(kernel_size % 2 != 0 && "Kernel size must be odd!");

    const double two_sigma_sq = 2.0 * sigma * sigma;
    const int kernel_radius = static_cast<int>(kernel_size / 2);
    
    double kernel_acc = 0.0;
    for (int i=0; i < static_cast<int>(kernel_size); ++i){
        double x = static_cast<double>(i - kernel_radius);
        kernel[i] = std::exp(- (x * x) / two_sigma_sq);
        kernel_acc += kernel[i];
    }

    for (size_t i=0; i < kernel_size; ++i){
        kernel[i] /= kernel_acc;
    }
}

void Convolve1D(const unsigned char* src, unsigned char* dst, int rows, 
        int cols, const double* kernel, size_t kernel_size, Axis axis)
{
    const int kernel_radius = static_cast<int>(kernel_size / 2); 
    int target_row, target_col;

    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            
            double accumulator = 0.0;

            for (int k = 0; k < static_cast<int>(kernel_size); ++k){
                const int offset = k - kernel_radius;
                
                if (axis == Axis::kHorizontal){
                    target_col = j + offset;
                    target_row = i;
                } else {
                    target_row = i + offset;
                    target_col = j;
                }

                const bool in_bounds = 
                        target_row >= 0 && target_row < rows &&
                        target_col >= 0 && target_col < cols;

                if (in_bounds) {
                    accumulator += kernel[k] * src[target_row * cols + target_col];
                }
            }
            
            dst[i * cols + j] = 
                static_cast<unsigned char>(
                        std::clamp(accumulator, 0.0, 255.0)
                    );
        }
    }
}

void GaussianFilterCPU(const unsigned char* src,
        unsigned char* dst, int rows, int cols,
        double* kernel, size_t kernel_size)
{
    std::vector<unsigned char> buffer(rows * cols);
    Convolve1D(src, buffer.data(), rows, cols, kernel, kernel_size, Axis::kHorizontal);
    Convolve1D(buffer.data(), dst, rows, cols, kernel, kernel_size, Axis::kVertical);
}

size_t KernelSizeForSigma(double sigma) {
    const size_t radius = static_cast<size_t>(std::ceil(3.0 * sigma));
    return 2 * radius + 1;
}

cv::Mat MakeLabeledPanel(const cv::Mat& gray, const std::string& label) {
    cv::Mat panel;
    cv::cvtColor(gray, panel, cv::COLOR_GRAY2BGR);
    cv::putText(panel, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
    return panel;
}

} // namespace

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

    cv::Mat image;
    double scale = 800.0 / std::max(raw.cols, raw.rows);
    cv::resize(raw, image, cv::Size(), scale, scale);

    const std::vector<double> sigmas = {1.0, 2.5, 5.0};

    std::vector<cv::Mat> panels;
    panels.push_back(MakeLabeledPanel(image, "original"));

    for (const double sigma : sigmas) {
        const size_t kernel_size = KernelSizeForSigma(sigma);
        std::vector<double> kernel(kernel_size);

        ComputeGaussianKernel1D(kernel.data(), kernel_size, sigma);

        cv::Mat blurred(image.rows, image.cols, CV_8UC1);
        GaussianFilterCPU(image.data, blurred.data, image.rows, image.cols,
                kernel.data(), kernel_size);

        std::ostringstream label;
        label << "sigma = " << std::fixed << std::setprecision(1) << sigma;
        panels.push_back(MakeLabeledPanel(blurred, label.str()));
    }

    cv::Mat mosaic = cv::Mat::zeros(
        image.rows * 2,
        image.cols * 2,
        CV_8UC3
    );

    for (size_t i = 0; i < panels.size(); ++i) {
        int row = static_cast<int>(i) / 2;
        int col = static_cast<int>(i) % 2;

        cv::Mat roi = mosaic(
            cv::Rect(
                col * image.cols,
                row * image.rows,
                image.cols,
                image.rows
            )
        );

        panels[i].copyTo(roi);
    }
    cv::imshow("Gaussian Blur - increasing sigma", mosaic);
    cv::waitKey(0);

    return 0;
}

