# Gaussian Blur Filter
Mathematically the same as convolving the image with a Gaussian function.
$$G(x, y) = G(x) \cdot G(y) = \frac{1}{2\pi\sigma^2}\exp{-\frac{x^2 + y^2}{2\sigma^2}}$$

Each pixel's new value is set to a weighted average of the neigboring pixels (weigthed by the Gaussian distribution)

![Gaussian](../assets/gaussian-blur.png)

## Implementation
Gaussian filter is **separable** meaning that it can be computed as a sequence of operations first in one dimension and then 
in the second (for 2d images). The implication of this are direct to our usecase of parallelizing on a GPU and also observable on 
the big O complexity analysis. Given a kernel of size $h_k  \times w_k$ the complexity:
1. Separable filter: $\mathcal{O}(w_k \cdot w_{img} \cdot h_{img}) + \mathcal{O}(h_k \cdot w_{img} \cdot h_{img})$
1. Non-Separable filter: $\mathcal{O}(w_k \cdot  h_k \cdot w_{img} \cdot h_{img})$

For simplicity reasons, this first Implementation will use a square kernel $k\times k$



# References
- https://en.wikipedia.org/wiki/Gaussian_blur
