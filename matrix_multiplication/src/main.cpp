#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../include/matmul.hpp"


void PrintSquareMatrix(std::vector<float>& mat)
{
    const size_t N = static_cast<size_t>(sqrtf(mat.size()));

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

int main(void)
{
    std::vector<float> mat1 = {
        1, 2, 3,
        4, 5, 6, 
        7, 8, 9
    };

    std::vector<float> dst(mat1.size());

    SquareMatMul(mat1, mat1, dst);

    std::cout << "Matrix multiplication finished!\n";

    PrintSquareMatrix(mat1);
    std::cout << "\tX\n";
    PrintSquareMatrix(mat1);
    std::cout << "\t=\n";
    PrintSquareMatrix(dst);
    
    cv::Mat A(3, 3, CV_32F, mat1.data());
    cv::Mat cvDst = A * A;

    std::cout << "\nOpenCV matrix multiplication:\n";
    std::cout << cvDst << '\n';

    return 0;
}
