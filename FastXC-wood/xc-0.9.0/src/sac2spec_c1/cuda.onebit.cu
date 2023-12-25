#include "cuda.onebit.cuh"

__global__ void onebit2DKernel(float *d_data, size_t pitch, size_t width,
                               size_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int idx = row * pitch + col;
        d_data[idx] = (d_data[idx] > 0.0f)   ? 1.0f
                      : (d_data[idx] < 0.0f) ? -1.0f
                                             : 0.0f;
    }
}

// do onebit time domain normalization
void onebit(float *d_data, int nseg, int proccnt)
{
    size_t pitch = nseg, width = nseg;
    size_t height = proccnt;
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    onebit2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height);
}