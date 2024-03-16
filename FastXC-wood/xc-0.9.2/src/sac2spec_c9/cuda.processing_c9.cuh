#ifndef _CU_PRE_PROCESS_H_
#define _CU_PRE_PROCESS_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.smooth.cuh"
#include "cuda.taper.cuh"

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int npts,
                int pitch, size_t proccnt, int taper_percentage);

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
               int filterCount, float delta, int proccnt, int nfft_1x, float maxval);

void freqWhiten_c9(cuComplex *d_spectrum_1,
                   cuComplex *d_spectrum_2,
                   cuComplex *d_spectrum_3,
                   float *d_weight_1, float *d_weight_2, float *d_weight_3,
                   float *d_tmp, int pitch, int width,
                   int height, int nsmooth, float df, float freq_low_limit, float freq_high_limit);

#endif