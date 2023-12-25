#ifndef _CU_WHITEN_C1_CUH
#define _CU_WHITEN_C1_CUH
#include <cuComplex.h>
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.smooth.cuh"
#include "cuda.taper.cuh"

void freqWhiten_c9(cuComplex *d_spectrum_1,
                   cuComplex *d_spectrum_2,
                   cuComplex *d_spectrum_3,
                   float *d_weight_1, float *d_weight_2, float *d_weight_3,
                   float *d_tmp, int pitch, int width,
                   int height, int nsmooth, float df, float freq_low_limit, float freq_high_limit, int filter_flag);
#endif