#ifndef _CU_PRE_PROCESS_C1_H_
#define _CU_PRE_PROCESS_C1_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.smooth.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.taper.cuh"

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int npts,
                int pitch, size_t proccnt, int taper_percentage);

void runabs_c1(float *d_timesignal,
               cuComplex *d_spectrum,
               float *d_filtered_timesignal,
               cuComplex *d_filtered_spectrum,
               float *d_weight, float *d_tmp,
               cufftHandle *planinv,
               cuComplex *d_filter_responses,
               float *freq_lows,
               int filterCount, float delta, int proccnt, 
               int nfft_1x, float maxval);

void freqWhiten(cuComplex *d_spectrum, float *d_weight, float *d_tmp, int pitch, int width,
                int height, int nsmooth, float df, float freq_low_limit, float freq_high_limit);

#endif