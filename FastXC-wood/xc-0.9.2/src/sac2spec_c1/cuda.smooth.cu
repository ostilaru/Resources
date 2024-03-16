#include "cuda.smooth.cuh"
#include <stdio.h>

__global__ void smoothTime2DKernel(float *d_out, int dpitch, float *d_tmp, int spitch, int width, int height, int maskwidth)
{
    double weight = 1.0 / (2 * maskwidth + 1);
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int idx = row * dpitch + col;
        double val = 0;
        int nstart = col - maskwidth / 2;

        for (int i = 0; i < maskwidth; i++)
        {
            if (nstart + i >= 0 && nstart + i < width)
            {
                val += d_tmp[row * spitch + nstart + i];
            }
        }
        d_out[idx] = weight * val;
    }
}

// width of npsmoothFreq2DKernel is 0.5 * nfft + 1
__global__ void smoothFreq2DKernel(float *d_out, int dpitch, float *d_tmp,
                                   int spitch, int width, int height,
                                   int maskwidth_max)
{
    // double weight = 1.0 / (2 * maskwidth + 1);
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process the first half of the spectrum
    if (col < width && row < height)
    {
        // int maskwidth = ((maskwidth_max -1) / width) * col + 1;
        int maskwidth = maskwidth_max * (col * col)/(width * width) + 1;
        double weight = 1.0 / (2 * maskwidth + 1);

        int idx = row * dpitch + col;
        double val = 0;
        int nstart = col - maskwidth / 2;

        for (int i = 0; i < maskwidth; i++)
        {
            int current_col = nstart + i;

            // if the left side of the mask is outside the spectrum
            if (current_col < 0)
            {
                current_col += maskwidth;
            }

            // if the right side of the mask is outside the spectrum
            if (current_col > width)
            {
                current_col -= maskwidth;
            }
            val += d_tmp[row * spitch + current_col];
        }

        float smooth_val = weight * val;
        d_out[idx] = smooth_val;
    }
}
