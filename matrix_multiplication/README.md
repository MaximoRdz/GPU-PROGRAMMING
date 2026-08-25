# Matrix Multiplication
C++ and Cuda implementations and optimization experiments.

main proble of matrix Multiplication via naive implementation:
once each thread is assigned a target dst (row, col) it has to perform an accumulation over the
multiplication of matrices `a` and `b`row and col elements (as given by the eqn). This accumultaiton in the gpu basically retrives from global memory `d_a[]` and `d_b[]`elements (in float precision 4 bytes) and writes in the result element `d_dst[]` another number. so two operations `*` and `+` for 8 bytes retrieved and another 4 bytes wrtten after. Really inefficient and bandwith bound for small size matrices.

## Naive implementation

<img src="./results/DatatypeBenchmark.png" width="700">
<img src="./results/DatatypeBenchmark_gpu.png" width="700">

## Tensor Cores and Matrix Multiplication

## Misc.
- **lambda function in c++**
```c++
[captures](parameters) {
    body
}
```
`[&]` allows the lambda function to access variables from the surrounding scope
and use them, only by reference!
# TODO
- [ ] performance for different data types and floating point precisions.
