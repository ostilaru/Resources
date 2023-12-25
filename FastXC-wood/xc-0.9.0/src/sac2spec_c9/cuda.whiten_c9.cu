#include "cuda.whiten_c9.cuh"

// frequency domain whitening
void freqWhiten_c9(cuComplex *d_spectrum_1,
                   cuComplex *d_spectrum_2,
                   cuComplex *d_spectrum_3,
                   float *d_weight_1, float *d_weight_2, float *d_weight_3,
                   float *d_tmp, int pitch, int width,
                   int height, int nsmooth, float df, float freq_low_limit, float freq_high_limit, int filter_flag)
{
    // set the width and height of data, only processing half of the spectrum
    size_t dpitch = pitch;
    size_t spitch = pitch;

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    amp2DKernel<<<dimgrd, dimblk>>>(d_weight_1, dpitch, d_spectrum_1, spitch, width, height);
    amp2DKernel<<<dimgrd, dimblk>>>(d_weight_2, dpitch, d_spectrum_2, spitch, width, height);
    amp2DKernel<<<dimgrd, dimblk>>>(d_weight_3, dpitch, d_spectrum_3, spitch, width, height);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_1, dpitch * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothFreq2DKernel<<<dimgrd, dimblk>>>(d_weight_1, dpitch, d_tmp, pitch, width, height, nsmooth);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_2, dpitch * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothFreq2DKernel<<<dimgrd, dimblk>>>(d_weight_2, dpitch, d_tmp, pitch, width, height, nsmooth);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_3, dpitch * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothFreq2DKernel<<<dimgrd, dimblk>>>(d_weight_3, dpitch, d_tmp, pitch, width, height, nsmooth);

    clampmin2DKernel<<<dimgrd, dimblk>>>(d_weight_1, pitch, width, height, MINVAL);
    clampmin2DKernel<<<dimgrd, dimblk>>>(d_weight_2, pitch, width, height, MINVAL);
    clampmin2DKernel<<<dimgrd, dimblk>>>(d_weight_3, pitch, width, height, MINVAL);

    // add to d_weight_1 as the final weight
    sum2DKernel<<<dimgrd, dimblk>>>(d_weight_1, dpitch, d_weight_2, spitch, width, height);
    sum2DKernel<<<dimgrd, dimblk>>>(d_weight_1, dpitch, d_weight_3, spitch, width, height);

    cdiv2DKernel<<<dimgrd, dimblk>>>(d_spectrum_1, dpitch, d_weight_1, spitch, width, height);
    cdiv2DKernel<<<dimgrd, dimblk>>>(d_spectrum_2, dpitch, d_weight_1, spitch, width, height);
    cdiv2DKernel<<<dimgrd, dimblk>>>(d_spectrum_3, dpitch, d_weight_1, spitch, width, height);

    // Apply spectra taper on [d_segspec]
    int idx1 = int(freq_low_limit * 0.667 / df);
    int idx2 = int(freq_low_limit / df);
    int idx3 = int(freq_high_limit / df);
    int idx4 = int(freq_high_limit * 1.333 / df);

    int np = 1; // window order

    if (filter_flag == 1)
    {
        // Using cosine window
        specTaper2DCosineKernel<<<dimgrd, dimblk>>>(d_spectrum_1, pitch, width, height, np, idx1, idx2, idx3, idx4);
        specTaper2DCosineKernel<<<dimgrd, dimblk>>>(d_spectrum_2, pitch, width, height, np, idx1, idx2, idx3, idx4);
        specTaper2DCosineKernel<<<dimgrd, dimblk>>>(d_spectrum_3, pitch, width, height, np, idx1, idx2, idx3, idx4);
    }
    else if (filter_flag == 2)
    {
        // Using Kaiser window
        specTaperKaiser2DKernel<<<dimgrd, dimblk>>>(d_spectrum_1, pitch, width, height, np, idx1, idx2, idx3, idx4, (double)BETA);
        specTaperKaiser2DKernel<<<dimgrd, dimblk>>>(d_spectrum_2, pitch, width, height, np, idx1, idx2, idx3, idx4, (double)BETA);
        specTaperKaiser2DKernel<<<dimgrd, dimblk>>>(d_spectrum_3, pitch, width, height, np, idx1, idx2, idx3, idx4, (double)BETA);
    }
}
