#include "cuda.processing_c1.cuh"

// pre-processing for sacdat: isnan, demean, detrend
void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int npts,
                int pitch, size_t proccnt, int taper_percentage)
{
    // check the nan/inf value of [d_sacdata]
    size_t width = npts;
    size_t height = proccnt;
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    dim3 dimgrd2, dimblk2;
    dimblk2.x = BLOCKMAX;
    dimblk2.y = 1;
    dimgrd2.x = 1;
    dimgrd2.y = height;

    isnan2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height);

    // demean. First calculate the mean value of each trace

    size_t dpitch = 1;
    size_t spitch = pitch;
    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    DimCompute(&dimgrd, &dimblk, width, height);
    rdc2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height,
                                    d_sum);

    // detrend. First calculate d_sum and d_isum

    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    isumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                              dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_isum, dpitch, d_sacdata, spitch, width, height);

    rtr2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height,
                                    d_sum, d_isum);

    // taper, taper percentage set in config.h
    timetaper2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, taper_percentage);
}

void runabs_onefreq_c1(cuComplex *d_filtered_spectrum,
                       float *d_filtered_timesignal,
                       cuComplex *d_response,
                       float *d_weight, float *d_tmp,
                       int nhalf_average_win,
                       float delta, int proccnt, int nfft_1x, float maxval,
                       cufftHandle *planinv)
{
  // set the width and height of data
  size_t twidth = nfft_1x;         // time domain width
  size_t fwidth = nfft_1x / 2 + 1; // frequency domain width
  size_t height = proccnt;
  size_t pitch = nfft_1x;

  // calculate the grid and block size for time domain and frequency domain
  dim3 tdimgrd, tdimblk;
  DimCompute(&tdimgrd, &tdimblk, twidth, height);

  dim3 fdimgrd, fdimblk;
  DimCompute(&fdimgrd, &fdimblk, fwidth, height);

  filterKernel<<<fdimgrd, fdimblk>>>(d_filtered_spectrum, d_response, pitch, fwidth, height);
  
  CUFFTCHECK(cufftExecC2R(*planinv, (cufftComplex *)d_filtered_spectrum, (cufftReal *)d_filtered_timesignal));
  InvNormalize2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal, pitch, twidth, height, delta);

  CUDACHECK(cudaMemcpy2D(d_weight, nfft_1x * sizeof(float), d_filtered_timesignal, nfft_1x * sizeof(float), nfft_1x * sizeof(float), proccnt, cudaMemcpyDeviceToDevice));

  abs2DKernel<<<tdimgrd, tdimblk>>>(d_weight, pitch, twidth, height);

  CUDACHECK(cudaMemcpy2D(d_tmp, pitch * sizeof(float), d_weight, pitch * sizeof(float), twidth * sizeof(float), height, cudaMemcpyDeviceToDevice));

  smoothTime2DKernel<<<tdimgrd, tdimblk>>>(d_weight, pitch, d_tmp, pitch, twidth, height, nhalf_average_win);

  // Avoid the minimum value is zero, old version is cunzero2D
  clampmin2DKernel<<<tdimgrd, tdimblk>>>(d_weight, pitch, twidth, height, MINVAL);

  div2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal, pitch, d_weight, pitch, twidth, height);

  isnan2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal, pitch, twidth, height);

  cutmax2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal, pitch, twidth, height, maxval);
}

// multi-frequency time domain normalization
void runabs_c1(float *d_timesignal,
               cuComplex *d_spectrum,
               float *d_filtered_timesignal,
               cuComplex *d_filtered_spectrum,
               float *d_weight, float *d_tmp,
               cufftHandle *planinv,
               cuComplex *d_filter_responses,
               float *freq_lows,
               int filterCount, float delta, int proccnt, 
               int nfft_1x, float maxval)
{
  size_t twidth = nfft_1x;
  size_t fwidth = nfft_1x * 0.5 + 1;
  size_t height = proccnt;
  size_t pitch = nfft_1x;

  // calculate the grid and block size for time domain and frequency domain
  dim3 dimgrd, dimblk;
  DimCompute(&dimgrd, &dimblk, twidth, height);

  // set d_timesignal to zero for the output
  CUDACHECK(cudaMemset(d_timesignal, 0, proccnt * nfft_1x * sizeof(float)));

  cuComplex *d_response = NULL;
  cudaMalloc((void **)&d_response, nfft_1x * sizeof(cuComplex));
  cudaMemset(d_response, 0, nfft_1x * sizeof(cuComplex));
  // time domain normalization on different frequency and add them together
  for (int i = 0; i < filterCount; i++)
  {
    // get the current filter response
    CUDACHECK(cudaMemcpy2D(d_response, nfft_1x * sizeof(cuComplex),
                           d_filter_responses + i * nfft_1x, nfft_1x * sizeof(cuComplex),
                           fwidth * sizeof(cuComplex), 1, cudaMemcpyDeviceToDevice));

    // refrence from Yao's code winsize = SampleF * EndT
    int nhalf_average_win = int(1.0 / (freq_lows[i] * delta )) + 1;

    CUDACHECK(cudaMemcpy2D(d_filtered_spectrum, nfft_1x * sizeof(cuComplex),
                           d_spectrum, nfft_1x * sizeof(cuComplex),
                           fwidth * sizeof(cuComplex), proccnt, cudaMemcpyDeviceToDevice));

    CUDACHECK(cudaMemset(d_filtered_timesignal, 0, proccnt * nfft_1x * sizeof(float)));

    CUDACHECK(cudaMemset(d_weight, 0, proccnt * nfft_1x * sizeof(float)));

    CUDACHECK(cudaMemset(d_tmp, 0, proccnt * nfft_1x * sizeof(float)));

    runabs_onefreq_c1(d_filtered_spectrum,
                      d_filtered_timesignal,
                      d_response,
                      d_weight, d_tmp,
                      nhalf_average_win,
                      delta, proccnt, nfft_1x,
                      maxval,planinv);

    // add different freq result [d_segdata_filted] to output [d_dataseg]
    sum2DKernel<<<dimgrd, dimblk>>>(d_timesignal, pitch, d_filtered_timesignal, pitch, twidth, height);
  }
}

void freqWhiten(cuComplex *d_spectrum, float *d_weight, float *d_tmp, int pitch, int width,
                int height, int nsmooth, float df, float freq_low_limit,float freq_high_limit)
{
    // set the width and height of data, only processing half of the spectrum
    size_t dpitch = pitch;
    size_t spitch = pitch;

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

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


