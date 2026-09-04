# Vision Models Misc

## Channels last for accelation on CPU

Memory has a great impact on deep learning models, nothing new there. Particularly, in vision models `Channels Last` memory layout is more favorable due to better data locality as we'll explain here.

As determined by the memory hardware design we are forced to store `nD` arrays in linear memory address space, in [matrix-multiplication](https://github.com/MaximoRdz/GPU-PROGRAMMING/blob/main/matrix_multiplication/README.md) experiments we saw, for example, the impact of row-major and col-major storing of matrices in memory. Let's now introduce two aspects:
1. **Physical order**: Layout fo data storage in physical memory. In vision models the common notation is `NCHW` and `NHWC` (channels first and last respectively), clearly N is the number of images, H height, W width and C channels.
1. **Logical Order**: Convention on how to describe tensor shape and stride. PyTorch's convention is NCHW, **no matter what the physical order is, tensor shape and stride will always be depicted in the order of `NCHW`**

[memory-format](https://pytorch.org/wp-content/uploads/2024/11/accelerating-pytorch-vision-models-with-channels-last-on-cpu-1.png)

The image is an example of tensor `[1, 3, 4, 4]` (NCHW as this is pytorch) on both channels first and channels last memory format (for people familiar with C++ sufferings this recalls AoS vs SoA)

### Memory Formats Propagation

General rule in pytorch: preserve the input tensor's memory format.

> Important: Can this be used to avoid overhead of kernels launched with the only purpose of reordering the dimensions to make the compatible with other functions (cuDNN or others)?

## REferences
- https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
