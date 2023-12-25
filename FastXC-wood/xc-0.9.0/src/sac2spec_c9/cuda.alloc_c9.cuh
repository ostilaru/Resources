#ifndef _ALLOC_CUH
#define _ALLOC_CUH
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include "in_out_node_c9.h"
#include "config.h"
#include "cuda.util.cuh"
#include "complex.h"
#include "path_node.h"
extern "C"
{
#include "util.h"
#include "complex.h"
}
void allocateCudaMemoryC9(int d_batch, int npts, int nstep_valid, int nfft_2x,
                          int do_runabs, int wh_flag,
                          float **d_timesignal_1, cuComplex **d_spectrum_1,
                          float **d_timesignal_2, cuComplex **d_spectrum_2,
                          float **d_timesignal_3, cuComplex **d_spectrum_3,
                          float **d_segment_timesignal_1, cuComplex **d_segment_spectrum_1,
                          float **d_segment_timesignal_2, cuComplex **d_segment_spectrum_2,
                          float **d_segment_timesignal_3, cuComplex **d_segment_spectrum_3,
                          float **d_filtered_segment_timesignal_1, cuComplex **d_filtered_segment_spectrum_1,
                          float **d_filtered_segment_timesignal_2, cuComplex **d_filtered_segment_spectrum_2,
                          float **d_filtered_segment_timesignal_3, cuComplex **d_filtered_segment_spectrum_3,
                          cuComplex **d_filter_response,
                          int filterCount,
                          float **d_weight_1, float **d_weight_2, float **d_weight_3,
                          float **d_tmp, double **d_sum, double **d_isum,
                          cufftHandle *planfwd, cufftHandle *planinv);

void freeMemory(cufftHandle planfwd, cufftHandle planinv,
                float **d_timesignal_1, cuComplex **d_spectrum_1,
                float **d_timesignal_2, cuComplex **d_spectrum_2,
                float **d_timesignal_3, cuComplex **d_spectrum_3,
                float **d_segment_timesignal_1, cuComplex **d_segment_spectrum_1,
                float **d_segment_timesignal_2, cuComplex **d_segment_spectrum_2,
                float **d_segment_timesignal_3, cuComplex **d_segment_spectrum_3,
                float **d_filtered_segment_timesignal_1, cuComplex **d_filtered_segment_spectrum_1,
                float **d_filtered_segment_timesignal_2, cuComplex **d_filtered_segment_spectrum_2,
                float **d_filtered_segment_timesignal_3, cuComplex **d_filtered_segment_spectrum_3,
                cuComplex **d_filter_response,
                float **d_weight_1, float **d_weight_2, float **d_weight_3,
                double **d_sum, double **d_isum, float **d_tmp,
                float **h_timesignal_1, complex **h_spectrum_1,
                float **h_timesignal_2, complex **h_spectrum_2,
                float **h_timesignal_3, complex **h_spectrum_3,
                InOutNodeC9 *pInOutList);
#endif