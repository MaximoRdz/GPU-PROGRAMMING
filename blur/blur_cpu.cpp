#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "blur_cpu.hpp"


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
