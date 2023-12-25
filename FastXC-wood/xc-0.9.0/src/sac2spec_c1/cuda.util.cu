#include "cuda.util.cuh"

const float RAMUPPERBOUND = 0.9;

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
    pdimblk->x = BLOCKX;
    pdimblk->y = BLOCKY;

    pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
    pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

void GpuFree(void **pptr)
{
    if (*pptr != NULL)
    {
        cudaFree((void **)*pptr);
        *pptr = NULL;
    }
}

size_t QueryAvailGpuRam(int deviceID)
{
    size_t freeram, totalram;
    cudaSetDevice(deviceID);
    cudaMemGetInfo(&freeram, &totalram);
    freeram *= RAMUPPERBOUND;

    const size_t gigabytes = 1L << 30;
    printf("Avail gpu ram on device%d: %.3f GB\n", deviceID,
           freeram * 1.0 / gigabytes);
    return freeram;
}
