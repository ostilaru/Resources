#ifndef _CUDA_RUNABS_C9_CUH
#define _CUDA_RUNABS_C9_CUH
#include "config.h"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.smooth.cuh"
#include "cuda.taper.cuh"
#include "cuda.util.cuh"
#include "cuda.fft_normalize.cuh"
#include <cuComplex.h>

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
               float df, float maxval);

#endif
