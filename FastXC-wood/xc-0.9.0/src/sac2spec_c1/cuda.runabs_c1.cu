#include "cuda.runabs_c1.cuh"

/* multi-frequency time domain normalization */
void runabs_onefreq_c1(cuComplex *d_filtered_spectrum,
                       float *d_filtered_timesignal,
                       cuComplex *d_response,
                       float *d_weight, float *d_tmp,
                       int nhalf_average_win,
                       float delta, int proccnt, int nfft_2x,
                       float df, float maxval,
                       cufftHandle *planinv)
{
  // set the width and height of data
  size_t twidth = nfft_2x;         // time domain width
  size_t fwidth = nfft_2x / 2 + 1; // frequency domain width
  size_t height = proccnt;
  size_t pitch = nfft_2x;

  // calculate the grid and block size for time domain and frequency domain
  dim3 tdimgrd, tdimblk;
  DimCompute(&tdimgrd, &tdimblk, twidth, height);

  dim3 fdimgrd, fdimblk;
  DimCompute(&fdimgrd, &fdimblk, fwidth, height);

  filterKernel<<<fdimgrd, fdimblk>>>(d_filtered_spectrum, d_response, pitch, fwidth, height);
  
  CUFFTCHECK(cufftExecC2R(*planinv, (cufftComplex *)d_filtered_spectrum, (cufftReal *)d_filtered_timesignal));
  InvNormalize2DKernel<<<tdimgrd, tdimblk>>>(d_filtered_timesignal, pitch, twidth, height, delta);

  CUDACHECK(cudaMemcpy2D(d_weight, nfft_2x * sizeof(float), d_filtered_timesignal, nfft_2x * sizeof(float), nfft_2x * sizeof(float), proccnt, cudaMemcpyDeviceToDevice));

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

  // set d_timesignal to zero for the output
  CUDACHECK(cudaMemset(d_timesignal, 0, proccnt * nfft_2x * sizeof(float)));

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

    CUDACHECK(cudaMemcpy2D(d_filtered_spectrum, nfft_2x * sizeof(cuComplex),
                           d_spectrum, nfft_2x * sizeof(cuComplex),
                           fwidth * sizeof(cuComplex), proccnt, cudaMemcpyDeviceToDevice));

    CUDACHECK(cudaMemset(d_filtered_timesignal, 0, proccnt * nfft_2x * sizeof(float)));

    CUDACHECK(cudaMemset(d_weight, 0, proccnt * nfft_2x * sizeof(float)));

    CUDACHECK(cudaMemset(d_tmp, 0, proccnt * nfft_2x * sizeof(float)));

    runabs_onefreq_c1(d_filtered_spectrum,
                      d_filtered_timesignal,
                      d_response,
                      d_weight, d_tmp,
                      nhalf_average_win,
                      delta, proccnt, nfft_2x,
                      df, maxval,
                      planinv);

    // add different freq result [d_segdata_filted] to output [d_dataseg]
    sum2DKernel<<<dimgrd, dimblk>>>(d_timesignal, pitch, d_filtered_timesignal, pitch, twidth, height);
  }
}
