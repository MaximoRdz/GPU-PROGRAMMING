#ifndef BLUR_CUDA_H
#define BLUR_CUDA_H

#include <cstddef>

void LaunchGaussianSmoothing(const unsigned char* h_src, unsigned char* h_dst,
        int ncols, int nrows, float* h_kernel, size_t kernel_size);

#endif
