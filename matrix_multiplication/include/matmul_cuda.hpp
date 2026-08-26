#ifndef CUDA_MATMUL_H
#define CUDA_MATMUL_H

#include <vector>

template <typename T>
void LaunchNaiveSqMatMulKernel(const std::vector<T>& h_a,
        const std::vector<T>& h_b, std::vector<T>& h_dst);

template <typename T>
void LaunchOptimizedSqMatMulKernel(const std::vector<T>& h_a,
        const std::vector<T>& h_b, std::vector<T>& h_dst);
#endif
