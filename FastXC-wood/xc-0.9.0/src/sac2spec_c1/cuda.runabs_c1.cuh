#ifndef _CUDA_RUNABS_C1_CUH
#define _CUDA_RUNABS_C1_CUH
#include "config.h"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.smooth.cuh"
#include "cuda.taper.cuh"
#include "cuda.util.cuh"
#include "cuda.fft_normalize.cuh"
#include <cuComplex.h>

void runabs_c1(float *d_timesignal,
               cuComplex *d_spectrum,
               float *d_filtered_timesignal,
               cuComplex *d_filtered_spectrum,
               float *d_weight, float *d_tmp,
               cufftHandle *planinv,
               cuComplex *d_filter_responses,
               float *freq_lows,
               int filterCount, float delta, int proccnt, int nfft_2x,
               float df, float maxval);

#endif