#ifndef _CUDA_XC_MONO_CUH
#define _CUDA_XC_MONO_CUH

#include "cuda.util.cuh"
#include "node_util.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void cmulmono2DKernel(cuComplex *d_spec,
                                 size_t srcpitch, size_t srcoffset,
                                 size_t stapitch, size_t staoffset,
                                 PAIRNODE *d_pairlist, size_t paircnt,
                                 cuComplex *d_segncf, size_t ncfpitch,
                                 int nspec);

__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height, int nstep);

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int nstep);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);

#endif