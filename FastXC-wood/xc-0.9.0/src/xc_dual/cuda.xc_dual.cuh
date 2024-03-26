#ifndef _CUDA_XC_DUAL_CUH
#define _CUDA_XC_DUAL_CUH

#include "cuda.util.cuh"
#include "node_util.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

// FIXME: in order to process large data, we need to add parameter batch_size
__global__ void cmuldual2DKernel(cuComplex *d_specsrcvec, size_t srcpitch,
                                 size_t srcoffset, cuComplex *d_specstavec,
                                 size_t stapitch, size_t staoffset,
                                 PAIRNODE *d_pairlist, size_t paircnt,
                                 cuComplex *d_segncfvec, size_t ncfpitch,
                                 int nspec, size_t current_batch_size);

__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height, int nstep);

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int nstep);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);

#endif