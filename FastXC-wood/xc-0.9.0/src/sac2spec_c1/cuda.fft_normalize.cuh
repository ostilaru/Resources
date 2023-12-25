#ifndef _CU_FFT_NORMALIZE_H_
#define _CU_FFT_NORMALIZE_H_
#include <cuComplex.h>
#include <stdio.h>

__global__ void FwdNormalize2DKernel(cuComplex *d_segspec, size_t pitch,
                                     size_t width, size_t height, float dt);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);
#endif