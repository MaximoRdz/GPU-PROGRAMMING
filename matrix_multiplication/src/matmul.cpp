/* #include <cassert> */
/* #include <cmath> */
/* #include <cstddef> */
/* #include <vector> */


/* template <typename T> */
/* void SquareMatMul(std::vector<T> &a, std::vector<T> &b, */
/*         std::vector<T> &dst) */
/* { */
/*     assert(a.size() == b.size() && "ERROR: Square Matrices size must be equal!"); */
/*     assert(a.size() == dst.size() && "ERROR: output matrix size must be equal!"); */

/*     const size_t N = static_cast<size_t>(std::sqrt(a.size())); */

/*     std::fill(dst.begin(), dst.end(), T{}); */

/*     for (size_t i = 0; i < N; ++i) { */
/*         for (size_t j = 0; j < N; ++j) { */
/*             for (size_t k = 0; k < N; ++k) { */
/*                 dst[i * N + j] += a[i * N + k] * b[k * N + j]; */
/*             } */
/*         } */
/*     } */
/* } */
