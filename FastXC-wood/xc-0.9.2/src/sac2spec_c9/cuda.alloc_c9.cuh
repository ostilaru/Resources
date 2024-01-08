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

size_t EstimateGpuBatchC9(size_t gpu_id, int npts, int nfft_1x, int nstep,
                          int filter_count,size_t wh_flag, size_t runabs_flag);

void allocateCudaMemoryC9(int d_batch, int npts, int nstep_valid, int nfft_1x,
                          int do_runabs, int wh_flag,
                          float **d_timesignal_1, cuComplex **d_spectrum_1,
                          float **d_timesignal_2, cuComplex **d_spectrum_2,
                          float **d_timesignal_3, cuComplex **d_spectrum_3,
                          float **d_segment_timesignal_1, cuComplex **d_segment_spectrum_1,
                          float **d_segment_timesignal_2, cuComplex **d_segment_spectrum_2,
                          float **d_segment_timesignal_3, cuComplex **d_segment_spectrum_3,
                          float **d_segment_timesignal_1_2x, cuComplex **d_segment_spectrum_1_2x,
                          float **d_segment_timesignal_2_2x, cuComplex **d_segment_spectrum_2_2x,
                          float **d_segment_timesignal_3_2x, cuComplex **d_segment_spectrum_3_2x,
                          float **d_filtered_segment_timesignal_1, cuComplex **d_filtered_segment_spectrum_1,
                          float **d_filtered_segment_timesignal_2, cuComplex **d_filtered_segment_spectrum_2,
                          float **d_filtered_segment_timesignal_3, cuComplex **d_filtered_segment_spectrum_3,
                          cuComplex **d_filter_responses,
                          int filterCount,
                          float **d_weight_1, float **d_weight_2, float **d_weight_3,
                          float **d_tmp, double **d_sum, double **d_isum,
                          cufftHandle *planfwd, cufftHandle *planinv,cufftHandle *planfwd_2x);

void freeMemory(cufftHandle planfwd, cufftHandle planinv,cufftHandle planfwd_2x,
                float **d_timesignal_1, cuComplex **d_spectrum_1,
                float **d_timesignal_2, cuComplex **d_spectrum_2,
                float **d_timesignal_3, cuComplex **d_spectrum_3,
                float **d_segment_timesignal_1, cuComplex **d_segment_spectrum_1,
                float **d_segment_timesignal_2, cuComplex **d_segment_spectrum_2,
                float **d_segment_timesignal_3, cuComplex **d_segment_spectrum_3,
                float **d_segment_timesignal_1_2x, cuComplex **d_segment_spectrum_1_2x,
                float **d_segment_timesignal_2_2x, cuComplex **d_segment_spectrum_2_2x,
                float **d_segment_timesignal_3_2x, cuComplex **d_segment_spectrum_3_2x,
                float **d_filtered_segment_timesignal_1, cuComplex **d_filtered_segment_spectrum_1,
                float **d_filtered_segment_timesignal_2, cuComplex **d_filtered_segment_spectrum_2,
                float **d_filtered_segment_timesignal_3, cuComplex **d_filtered_segment_spectrum_3,
                cuComplex **d_filter_responses,
                float **d_weight_1, float **d_weight_2, float **d_weight_3,
                double **d_sum, double **d_isum, float **d_tmp,
                float **h_timesignal_1, complex **h_spectrum_1,
                float **h_timesignal_2, complex **h_spectrum_2,
                float **h_timesignal_3, complex **h_spectrum_3,
                InOutNodeC9 *pInOutList);
#endif