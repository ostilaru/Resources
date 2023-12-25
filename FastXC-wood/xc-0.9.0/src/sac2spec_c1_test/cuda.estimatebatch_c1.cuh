#ifndef _CU_ESTIMATE_BATCH_C1_H_
#define _CU_ESTIMATE_BATCH_C1_H_
#include <cuComplex.h>
#include "cuda.util.cuh"

size_t EstimateGpuBatchC1(int gpu_id, int npts, int nfft_2x, int nstep,int filter_count,
                          size_t wh_flag, size_t runabs_flag);

#endif