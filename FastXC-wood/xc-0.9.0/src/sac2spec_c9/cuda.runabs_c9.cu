#include "cuda.runabs_c9.cuh"
#include "complex.h"

/* multi-frequency time domain normalization */
void runabs_onefreq_c9(cuComplex *d_filtered_spectrum_1,
                       cuComplex *d_filtered_spectrum_2,
                       cuComplex *d_filtered_spectrum_3,
                       float *d_filtered_timesignal_1,
                       float *d_filtered_timesignal_2,
                       float *d_filtered_timesignal_3,
                       cuComplex *d_response,
                       float *d_weight_1,
                       float *d_weight_2,
                       float *d_weight_3,
                       float *d_tmp,
                       int nhalf_average_win,
                       float delta, int proccnt, int nfft_2x,
                       float df, float maxval,
                       cufftHandle *planinv)
{
    printf("Hello world!\n");
    // set the width and height of data
    size_t twidth = nfft_2x;         // time domain width
    size_t fwidth = nfft_2x / 2 + 1; // frequency domain width
    size_t height = proccnt;
    size_t dpitch = nfft_2x; // destination pitch
    size_t spitch = nfft_2x; // source pitch
    size_t pitch = nfft_2x;

    // calculate the grid and block size for time domain and frequency domain
    dim3 tdimgrd, tdimblk;
    DimCompute(&tdimgrd, &tdimblk, twidth, height);

    dim3 fdimgrd, fdimblk;
    DimCompute(&fdimgrd, &fdimblk, fwidth, height);

    filterKernel<<<fdimgrd, fdimblk>>>(d_filtered_spectrum_1, d_response, pitch, fwidth, height);
    filterKernel<<<fdimgrd, fdimblk>>>(d_filtered_spectrum_2, d_response, pitch, fwidth, height);
    filterKernel<<<fdimgrd, fdimblk>>>(d_filtered_spectrum_3, d_response, pitch, fwidth, height);

    CUFFTCHECK(cufftExecC2R(*planinv, (cufftComplex *)d_filtered_spectrum_1, (cufftReal *)d_filtered_timesignal_1));
    InvNormalize2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_1, pitch, twidth, height, delta);

    CUFFTCHECK(cufftExecC2R(*planinv, (cufftComplex *)d_filtered_spectrum_2, (cufftReal *)d_filtered_timesignal_2));
    InvNormalize2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_2, pitch, twidth, height, delta);

    CUFFTCHECK(cufftExecC2R(*planinv, (cufftComplex *)d_filtered_spectrum_3, (cufftReal *)d_filtered_timesignal_3));
    InvNormalize2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_3, pitch, twidth, height, delta);

    CUDACHECK(cudaMemcpy2D(d_weight_1, nfft_2x * sizeof(float),
                           d_filtered_timesignal_1, nfft_2x * sizeof(float),
                           nfft_2x * sizeof(float), proccnt, cudaMemcpyDeviceToDevice));

    CUDACHECK(cudaMemcpy2D(d_weight_2, nfft_2x * sizeof(float),
                           d_filtered_timesignal_2, nfft_2x * sizeof(float),
                           nfft_2x * sizeof(float), proccnt, cudaMemcpyDeviceToDevice));

    CUDACHECK(cudaMemcpy2D(d_weight_3, nfft_2x * sizeof(float),
                           d_filtered_timesignal_3, nfft_2x * sizeof(float),
                           nfft_2x * sizeof(float), proccnt, cudaMemcpyDeviceToDevice));

    abs2DKernel<<<tdimgrd, tdimblk>>>(d_weight_1, pitch, twidth, height);
    abs2DKernel<<<tdimgrd, tdimblk>>>(d_weight_2, pitch, twidth, height);
    abs2DKernel<<<tdimgrd, tdimblk>>>(d_weight_3, pitch, twidth, height);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_1, dpitch * sizeof(float), twidth * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothTime2DKernel<<<tdimgrd, tdimblk>>>(d_weight_1, dpitch, d_tmp, pitch, twidth, height, nhalf_average_win);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_2, dpitch * sizeof(float), twidth * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothTime2DKernel<<<tdimgrd, tdimblk>>>(d_weight_2, dpitch, d_tmp, pitch, twidth, height, nhalf_average_win);

    CUDACHECK(cudaMemcpy2D(d_tmp, dpitch * sizeof(float), d_weight_3, dpitch * sizeof(float), twidth * sizeof(float), height, cudaMemcpyDeviceToDevice));
    smoothTime2DKernel<<<tdimgrd, tdimblk>>>(d_weight_3, dpitch, d_tmp, pitch, twidth, height, nhalf_average_win);

    // add to d_weight_1 as the final weight
    sum2DKernel<<<tdimgrd, tdimblk>>>(d_weight_1, dpitch, d_weight_2, spitch, twidth, height);
    sum2DKernel<<<tdimgrd, tdimblk>>>(d_weight_1, dpitch, d_weight_3, spitch, twidth, height);

    // Avoid the minimum value is zero, old version is cunzero2D
    clampmin2DKernel<<<tdimgrd, tdimblk>>>(d_weight_1, dpitch, twidth, height, MINVAL);

    div2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_1, dpitch, d_weight_1, spitch, twidth, height);
    div2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_2, dpitch, d_weight_1, spitch, twidth, height);
    div2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_3, dpitch, d_weight_1, spitch, twidth, height);

    isnan2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_1, pitch, twidth, height);
    isnan2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_2, pitch, twidth, height);
    isnan2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_3, pitch, twidth, height);

    cutmax2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_1, pitch, twidth, height, maxval);
    cutmax2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_2, pitch, twidth, height, maxval);
    cutmax2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal_3, pitch, twidth, height, maxval);
}

// multi-frequency time domain normalization
void runabs_c9(float *d_timesignal_1,
               float *d_timesignal_2,
               float *d_timesignal_3,
               cuComplex *d_spectrum_1,
               cuComplex *d_spectrum_2,
               cuComplex *d_spectrum_3,
               float *d_filtered_timesignal_1,
               float *d_filtered_timesignal_2,
               float *d_filtered_timesignal_3,
               cuComplex *d_filtered_spectrum_1,
               cuComplex *d_filtered_spectrum_2,
               cuComplex *d_filtered_spectrum_3,
               float *d_weight_1, float *d_weight_2, float *d_weight_3,
               float *d_tmp,
               cufftHandle *planinv,
               cuComplex *d_filter_responses,
               float *freq_lows,
               int filterCount, float delta, int proccnt, int nfft_2x,
               float df, float maxval)
{
    size_t twidth = nfft_2x;
    size_t fwidth = nfft_2x * 0.5 + 1;
    size_t height = proccnt;
    size_t pitch = nfft_2x;

    // calculate the grid and block size for time domain and frequency domain
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, twidth, height);

    // set d_timesignal_* to zero for the output
    CUDACHECK(cudaMemset(d_timesignal_1, 0, proccnt * nfft_2x * sizeof(float)));
    CUDACHECK(cudaMemset(d_timesignal_2, 0, proccnt * nfft_2x * sizeof(float)));
    CUDACHECK(cudaMemset(d_timesignal_3, 0, proccnt * nfft_2x * sizeof(float)));

    cuComplex *d_response = NULL;
    cudaMalloc((void **)&d_response, nfft_2x * sizeof(cuComplex));
    cudaMemset(d_response, 0, nfft_2x * sizeof(cuComplex));

    // time domain normalization on different frequency and add them together
    for (int i = 0; i < filterCount; i++)
    {
        // get the current filter response
        CUDACHECK(cudaMemcpy2D(d_response, nfft_2x * sizeof(cuComplex),
                               d_filter_responses + i * nfft_2x, nfft_2x * sizeof(cuComplex),
                               fwidth * sizeof(cuComplex), 1, cudaMemcpyDeviceToDevice));

        // refrence from Yao's code winsize = SampleF * EndT
        int nhalf_average_win = int(1.0 / (freq_lows[i] * delta)) + 1;

        CUDACHECK(cudaMemcpy2D(d_filtered_spectrum_1, nfft_2x * sizeof(cuComplex),
                               d_spectrum_1, nfft_2x * sizeof(cuComplex),
                               fwidth * sizeof(cuComplex), proccnt, cudaMemcpyDeviceToDevice));

        CUDACHECK(cudaMemcpy2D(d_filtered_spectrum_2, nfft_2x * sizeof(cuComplex),
                               d_spectrum_2, nfft_2x * sizeof(cuComplex),
                               fwidth * sizeof(cuComplex), proccnt, cudaMemcpyDeviceToDevice));

        CUDACHECK(cudaMemcpy2D(d_filtered_spectrum_3, nfft_2x * sizeof(cuComplex),
                               d_spectrum_3, nfft_2x * sizeof(cuComplex),
                               fwidth * sizeof(cuComplex), proccnt, cudaMemcpyDeviceToDevice));

        CUDACHECK(cudaMemset(d_filtered_timesignal_1, 0, proccnt * nfft_2x * sizeof(float)));
        CUDACHECK(cudaMemset(d_filtered_timesignal_2, 0, proccnt * nfft_2x * sizeof(float)));
        CUDACHECK(cudaMemset(d_filtered_timesignal_3, 0, proccnt * nfft_2x * sizeof(float)));

        CUDACHECK(cudaMemset(d_weight_1, 0, proccnt * nfft_2x * sizeof(float)));
        CUDACHECK(cudaMemset(d_weight_2, 0, proccnt * nfft_2x * sizeof(float)));
        CUDACHECK(cudaMemset(d_weight_3, 0, proccnt * nfft_2x * sizeof(float)));

        CUDACHECK(cudaMemset(d_tmp, 0, proccnt * nfft_2x * sizeof(float)));

        runabs_onefreq_c9(d_filtered_spectrum_1,
                          d_filtered_spectrum_2,
                          d_filtered_spectrum_3,
                          d_filtered_timesignal_1,
                          d_filtered_timesignal_2,
                          d_filtered_timesignal_3,
                          d_response,
                          d_weight_1, d_weight_2, d_weight_3,
                          d_tmp,
                          nhalf_average_win,
                          delta, proccnt, nfft_2x,
                          df, maxval,
                          planinv);

        dim3 tdimgrd, tdimblk;
        DimCompute(&tdimgrd, &tdimblk, twidth, height);

        // add different freq result [d_segdata_filted] to output [d_dataseg]
        sum2DKernel<<<dimgrd, dimblk>>>(d_timesignal_1, pitch, d_filtered_timesignal_1, pitch, twidth, height);
        sum2DKernel<<<dimgrd, dimblk>>>(d_timesignal_2, pitch, d_filtered_timesignal_2, pitch, twidth, height);
        sum2DKernel<<<dimgrd, dimblk>>>(d_timesignal_3, pitch, d_filtered_timesignal_3, pitch, twidth, height);
    }
}
