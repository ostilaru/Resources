#ifndef _CU_ALLOC_C1_CUH
#define _CU_ALLOC_C1_CUH
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include "in_out_node_c1.h"
#include "config.h"
#include "cuda.util.cuh"
#include "complex.h"
#include "path_node.h"
extern "C"
{
#include "util.h"
#include "complex.h"
}

void allocateCudaMemoryC1(int d_batch, int npts, int nstep_valid, int nfft_2x,
                          int do_runabs, int wh_flag, 
                          float **d_timesignal,
                          cuComplex **d_spectrum,
                          float **d_segment_timesignal,
                          cuComplex **d_segment_spectrum,
                          float **d_segment_timesignal_2x,
                          cuComplex **d_segment_spectrum_2x,
                          float **d_filtered_segment_timesignal,
                          cuComplex **d_filtered_segment_spectrum,
                          cuComplex **d_filter_response,
                          int filterCount,
                          float **d_weight, float **d_tmp,
                          double **d_sum, double **d_isum,
                          cufftHandle *planfwd, cufftHandle *planinv,cufftHandle *planfwd_2x);

void freeMemory(cufftHandle planfwd, cufftHandle planinv,
                float **d_timesignal,
                cuComplex **d_spectrum,
                float **d_segment_timesignal,
                cuComplex **d_segment_spectrum,
                float **d_segment_timesignal_2x,
                cuComplex **d_segment_spectrum_2x,
                float **d_filtered_segment_timesignal,
                cuComplex **d_filtered_segment_spectrum,
                cuComplex **d_filter_responses,
                float **d_weight, float **d_tmp,
                double **d_sum, double **d_isum,
                float **h_timesignal, complex **h_spectrum,
                InOutNodeC1 *pInOutList);
#endif