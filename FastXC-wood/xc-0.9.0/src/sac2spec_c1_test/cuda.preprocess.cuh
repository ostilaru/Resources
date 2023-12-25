#ifndef _CU_PRE_PROCESS_H_
#define _CU_PRE_PROCESS_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.taper.cuh"

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int npts,
                int pitch, size_t proccnt, int taper_percentage);

#endif