#include "cuda.whiten_c1.cuh"

// frequency domain whitening
void freqWhiten(cuComplex *d_spectrum, float *d_weight, float *d_tmp, int pitch, int width,
                int height, int nsmooth, float df, float freq_low_limit,float freq_high_limit, int filter_flag)
{
    // set the width and height of data, only processing half of the spectrum
    size_t dpitch = pitch;
    size_t spitch = pitch;

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    // if (filter_flag == 1)
    // {
    //     // Using cosine window
    //     specTaper2DCosineKernel<<<dimgrd, dimblk>>>(d_spectrum, pitch, width, height, np, idx1, idx2, idx3, idx4);
    // }
    // else if (filter_flag == 2)
    // {
    //     // Using Kaiser window
    //     specTaperSinc2DKernel<<<dimgrd, dimblk>>>(d_spectrum, pitch, width, height, idx1, idx2, idx3, idx4);
    // }

    amp2DKernel<<<dimgrd, dimblk>>>(d_weight, dpitch, d_spectrum, spitch, width, height);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight, dpitch * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyDeviceToDevice));

    smoothFreq2DKernel<<<dimgrd, dimblk>>>(d_weight, dpitch, d_tmp, pitch, width, height, nsmooth);

    clampmin2DKernel<<<dimgrd, dimblk>>>(d_weight, pitch, width, height, MINVAL);

    cdiv2DKernel<<<dimgrd, dimblk>>>(d_spectrum, dpitch, d_weight, spitch, width, height);

    // Apply spectra taper on [d_segspec]
    int idx1 = int(freq_low_limit * 0.667 / df);
    int idx2 = int(freq_low_limit / df);
    int idx3 = int(freq_high_limit / df);
    int idx4 = int(freq_high_limit * 1.333 / df);
    int np = 1; // window order
    specTaper2DCosineKernel<<<dimgrd, dimblk>>>(d_spectrum, pitch, width, height, np, idx1, idx2, idx3, idx4);
}
