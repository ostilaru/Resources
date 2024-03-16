#include "cuda.util.cuh"

const float RAMUPPERBOUND = 0.9;

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX;
  pdimblk->y = BLOCKY;

  pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
  pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

size_t QueryAvailGpuRam(size_t deviceID)
{
  size_t freeram, totalram;
  cudaSetDevice(deviceID);
  CUDACHECK(cudaMemGetInfo(&freeram, &totalram));
  freeram *= RAMUPPERBOUND;

  const size_t gigabytes = 1L << 30;
  printf("Avail gpu ram: %.3f GB\n", freeram * 1.0 / gigabytes);
  return freeram;
}

size_t EstimateGpuBatch(size_t gpu_id, size_t fixedRam, size_t unitram,
                        int numType, int rank, int *n, int *inembed,
                        int istride, int idist, int *onembed, int ostride,
                        int odist, cufftType *typeArr)
{
  int i;
  size_t d_batch = 0;
  size_t availram = QueryAvailGpuRam(gpu_id);

  size_t reqram = fixedRam;
  if (reqram > availram)
  {
    fprintf(stderr, "Not enough gpu ram required:%lu, gpu reamin ram: %lu\n",
            reqram, availram);
    exit(1);
  }

  size_t tmpram = 0;
  size_t fftram = 0;

  // calculate the memory used in FFT, changed 202230925
  for (i = 0; i < numType; i++)
  {
    cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, typeArr[i], 1, &tmpram);
    fftram += tmpram;
  }

  while (reqram <= availram)
  {
    //for (i = 0; i < numType; i++)
    //{
    //  cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, typeArr[i], d_batch, &tmpram);
    //  reqram += tmpram;
    //}
    reqram += d_batch * (unitram+fftram);
    d_batch++;
  }
    const size_t gigabytes = 1L << 30;
   printf("Used gpu ram: %.3f GB\n", reqram * 1.0 / gigabytes);
  d_batch--;
  if (d_batch == 0)
  {
    fprintf(stderr, "Not enough gpu ram required:%lu, gpu remain mem: %lu\n",
            reqram, availram);
    exit(1);
  }
  return d_batch;
}

void CufftPlanAlloc(cufftHandle *pHandle, int rank, int *n, int *inembed,
                    int istride, int idist, int *onembed, int ostride,
                    int odist, cufftType type, int batch)
{
  // create cufft plan
  CUFFTCHECK(cufftPlanMany(pHandle, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch));
}

void GpuMalloc(void **pptr, size_t sz) { CUDACHECK(cudaMalloc(pptr, sz)); }

void GpuCalloc(void **pptr, size_t sz)
{
  CUDACHECK(cudaMalloc(pptr, sz));

  CUDACHECK(cudaMemset(*pptr, 0, sz));
}

void GpuFree(void **pptr)
{
  CUDACHECK(cudaFree(*pptr));
  *pptr = NULL;
}
