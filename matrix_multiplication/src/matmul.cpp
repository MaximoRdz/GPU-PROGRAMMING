#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>


void SquareMatMul(std::vector<float> &a, std::vector<float> &b,
        std::vector<float> &dst)
{
    assert(a.size() == b.size() && "ERROR: Square Matrices size must be equal!");
    assert(a.size() == dst.size() && "ERROR: output matrix size must be equal!");

    const size_t N = static_cast<size_t>(sqrtf(a.size()));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                dst[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}
