#include <iostream>
#include <vector>

void PrintRowMajor(const std::vector<float>& A, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            std::cout << A[i * cols + j] << " ";
        std::cout << "\n";
    }
}

void PrintColMajor(const std::vector<float>& A, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            std::cout << A[i + j * rows] << " ";
        std::cout << "\n";
    }
}

void RunRowColMajorExample() {
    size_t M = 2, N = 3;
    std::vector<float> rowMajorA = {1,2,3,4,5,6};
    std::vector<float> colMajorA = {1,4,2,5,3,6};

    std::cout << "A (row-major storage), read as row-major:\n";
    PrintRowMajor(rowMajorA, M, N);   

    std::cout << "\nSame row-major bytes, read as col-major with rows=N,cols=M "
                 "(this gives A transposed):\n";
    PrintColMajor(rowMajorA, N, M);  

    std::cout << "\nA (col-major storage), read as col-major:\n";
    PrintColMajor(colMajorA, M, N); 

    std::cout << "\nSame col-major bytes, read as row-major with rows=M,cols=N "
                 "(this gives A transposed):\n";
    PrintColMajor(rowMajorA, M, N);  
}
