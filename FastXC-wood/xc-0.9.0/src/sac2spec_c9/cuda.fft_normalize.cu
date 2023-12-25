#include <cuComplex.h>


// FFT Nomalize add by wangjx@2023-05-21
__global__ void FwdNormalize2DKernel(cuComplex *d_segspec, size_t pitch,
                                     size_t width, size_t height, float dt)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    double weight = dt;
    if (row < height && col < width)
    {
        size_t idx = row * pitch + col;
        d_segspec[idx].x *= weight;
        d_segspec[idx].y *= weight;
    }
}

// IFFT Nomalize add by wangjx@2023-05-21
__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    double weight = 1.0 / (width * dt);
    if (row < height && col < width)
    {
        size_t idx = row * pitch + col;
        d_segdata[idx] *= weight;
    }
}