#ifndef __CU_SMOOTH_H__
#define __CU_SMOOTH_H__

__global__ void smoothTime2DKernel(float *d_out, int dpitch, float *d_tmp,
                                   int spitch, int width, int height,
                                   int maskwidth);

__global__ void smoothFreq2DKernel(float *d_out, int dpitch, float *d_tmp,
                                   int spitch, int width, int height,
                                   int maskwidth);

#endif