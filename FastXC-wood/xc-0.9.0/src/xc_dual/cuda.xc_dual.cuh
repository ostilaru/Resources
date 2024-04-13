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

// -------------------- Hilbert Transform --------------------
__global__ void Divide(cufftComplex *dev_complex, const int LENGTH, const int num_signals);
__global__ void Hilbert(cufftComplex *dev_complex, const int CPX_LEN, const int LENGTH, const int num_signals);
void HibertTransform(cufftComplex *dev_complex, const int LENGTH, const int num_signals, dim3 G_SIZE, dim3 B_SIZE, cufftHandle c2c_plan);
void HibertTransform_Spectrum(cufftComplex *dev_complex, const int LENGTH, const int num_signals, dim3 G_SIZE, dim3 B_SIZE, cufftHandle c2c_plan);

// ------------------- cuda PWS-Stack -------------------
__global__ void cudaMeanReal(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp);
__global__ void cudaMeanReal_abs(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp);
__global__ void cudaMeanImg(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_img);
__global__ void cudaMean(cufftComplex *hilbert_complex, int length, int num_signals, cufftComplex *mean);
__global__ void cudaABS(cufftComplex *hilbert_complex, int length, int num_signals, float *abs);
__global__ void cudaDivide(cufftComplex *hilbert_complex, int length, int num_signals, float *abs);
__global__ void cudaMultiply(float *num1, float *num2, float *res, int length);
__device__ void atomicMaxFloat(float* address, float val);
__global__ void findMaxValKernel(float *num, float *maxVal, int length);
__global__ void cudaNormalize(float *num, int length, float *maxVal);
void cudaPwsStack(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp, float *weighted_amp, dim3 G_SIZE, dim3 B_SIZE);

// ------------------- host PWS-Stack -------------------
void calculateMeanReal(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp);
void calculateMeanReal_abs(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp);
void calculateMeanImg(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_img);
void calculateMean(cufftComplex *hilbert_complex, size_t length, int num_signals, cufftComplex *mean);
void calculateABS(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs);
void calculateDivideMean(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs, cufftComplex *divide_mean);
void calculateMultiply(float *num1, float *num2, float *res, size_t length);
void normalize(float *num, size_t length);
void pws_stack(cufftComplex *hilbert_complex, const int length, const int num_signals, float *mean_amp, float *weighted_amp);

// ------------------- Test -------------------
__global__ void Print_Complex(cufftComplex *dev_complex, const int LENGTH);
__global__ void initComplexArray(cuComplex *arr, int n);

#endif