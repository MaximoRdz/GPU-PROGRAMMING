# Gray scale conversion
CPU vs GPU comparison.

## Overview
1. allocate memory on gpu
1. transfer data from cpu to gpu
1. launch a kernel
1. transfer back results from gpu to cpu
1. free gpu memory


1. **OpenCV** using `cv::cvtColor`
2. **`myGrayImage`** — a straightforward CPU implementation using `image.at<cv::Vec3b>()`
3. **`myGrayImageOpt`** — an optimized CPU implementation using row pointers
4. **CUDA** — a GPU implementation using a custom CUDA kernel

The grayscale conversion uses the standard weighted RGB formula:
```text
Gray = 0.299 R + 0.587 G + 0.114 B
```
### Execution time

| Implementation | Average time per call |
|---|---:|
| OpenCV `cv::cvtColor` | **89.88 µs** |
| `myGrayImage` | **5275.35 µs** |
| `myGrayImageOpt` | **1350.19 µs** |
| CUDA kernel only | **14.6566 µs** |
| CUDA total, including transfers | **265.452 µs** |

The CUDA implementation is therefore very fast when considering only the GPU kernel, but the cost of transferring the image between CPU and GPU makes the complete CUDA operation slower than OpenCV for this relatively small image.


### `myGrayImageOpt`

The optimized CPU implementation obtains a pointer to each row:

```cpp
const uchar* src_row = image.ptr<uchar>(r);
uchar* dst_row = output_image.ptr<uchar>(r);
```

Pixels can then be accessed directly using their byte offsets:

```cpp
uchar B = src_row[c * 3 + 0];
uchar G = src_row[c * 3 + 1];
uchar R = src_row[c * 3 + 2];
```

Its measured performance was:

**1350.19 µs per call**

This is a substantial improvement over the original implementation:

```text
5275.35 / 1350.19 ≈ 3.91×
```
## 4. CUDA performance

The CUDA implementation separates two different costs:

### Kernel-only time

The GPU kernel itself takes:

**14.6566 µs per call**

Compared with OpenCV:

```text
89.88 / 14.6566 ≈ 6.13×
```

Thus, the actual grayscale computation on the GPU is approximately **6.1× faster than OpenCV** in this test.

This is the main advantage of GPU parallelism: each pixel can be processed independently, allowing thousands of GPU threads to work on different pixels simultaneously.

The kernel uses a 2D grid of 16×16 thread blocks:

```cpp
dim3 blockDim(16, 16);
```

Each CUDA thread processes one pixel.

> **A faster kernel does not necessarily mean a faster application. The cost of moving data to and from the GPU must also be considered.**

## Misc.
upon observartion of the max abs diff of the different implemente methods it migth be possible
that for gpu approach we haven't take into account internal padding of the `cv::Mat`data class
accounting for really silent bug, difference is not large, at least for this tested image but
it might compound downstream in the pipeline.

- `.at<uchar>` and `.ptr<uchar>` internally account for it
- gpu implementation does not: 
        `int idx = (row * width + col) * channels;`

> **UPDATE**: this was not an issue related to the max diff!

> **UPDATE 2**: the error came from the fact that openCV works with BGR by default and in the cuda
> approach I misordered the channels...


