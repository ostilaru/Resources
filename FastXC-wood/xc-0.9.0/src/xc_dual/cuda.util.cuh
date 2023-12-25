#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH

#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BLOCKX 32
#define BLOCKY 32

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUFFTCHECK(cmd)                                                        \
  do {                                                                         \
    cufftResult_t e = cmd;                                                     \
    if (e != CUFFT_SUCCESS) {                                                  \
      printf("Failed: CuFFT error %s:%d %d\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

size_t QueryAvailGpuRam(size_t gpu_id);

size_t EstimateGpuBatch(size_t, size_t, size_t, int, int, int *, int *, int,
                        int, int *, int, int, cufftType *);

void DimCompute(dim3 *, dim3 *, size_t, size_t);

void CufftPlanAlloc(cufftHandle *, int, int *, int *, int, int, int *, int, int,
                    cufftType, int);
void GpuMalloc(void **, size_t);
void GpuCalloc(void **, size_t);
void GpuFree(void **);

#endif