#ifndef BLUR_CPU_H
#define BLUR_CPU_H

#include <cstddef>


enum class Axis { kHorizontal, kVertical };

void ComputeGaussianKernel1D(double* kernel, size_t kernel_size, double sigma);

void Convolve1D(const unsigned char* src, unsigned char* dst, int rows, 
        int cols, const double* kernel, size_t kernel_size, Axis axis);

void GaussianFilterCPU(const unsigned char* src,
        unsigned char* dst, int rows, int cols,
        double* kernel, size_t kernel_size);

size_t KernelSizeForSigma(double sigma);

#endif
