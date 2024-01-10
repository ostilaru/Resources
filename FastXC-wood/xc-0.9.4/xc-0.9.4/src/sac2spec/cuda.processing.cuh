#ifndef _CU_PRE_PROCESS_H_
#define _CU_PRE_PROCESS_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.smooth.cuh"
#include "cuda.taper.cuh"

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int pitch, size_t proccnt, int taper_percentage);

void runabs(float *d_sacdata, cuComplex *d_spectrum,
            float *d_filtered_sacdata, cuComplex *d_filtered_spectrum,
            cuComplex *d_responses, float *d_tmp,
            float *d_weight, float *d_tmp_weight,
            cufftHandle *planinv, float *freq_lows,
            int filterCount, float delta, int proc_batch, int num_ch, int pitch, float maxval);

void freqWhiten(cuComplex *d_spectrum,
                float *d_weight, float *d_tmp_weight, float *d_tmp,
                int num_ch, int pitch, int proc_batch,
                int nsmooth, int idx1, int idx2, int idx3, int idx4);

#endif