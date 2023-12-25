#ifndef _CU_ESTIMATE_BATCH_C9_H_
#define _CU_ESTIMATE_BATCH_C9_H_
#include <cuComplex.h>
#include "cuda.util.cuh"

size_t EstimateGpuBatchC9(int gpu_id, int npts, int nfft_2x, int nstep,
                          size_t wh_flag, size_t runabs_flag);

#endif