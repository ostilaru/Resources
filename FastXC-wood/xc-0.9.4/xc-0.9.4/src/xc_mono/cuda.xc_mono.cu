#include "cuda.xc_mono.cuh"

__global__ void cmulmono2DKernel(cuComplex *d_spec,
                                 size_t srcpitch, size_t srcoffset,
                                 size_t stapitch, size_t staoffset,
                                 PAIRNODE *d_pairlist, size_t paircnt,
                                 cuComplex *d_segncf, size_t ncfpitch,
                                 int nspec)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < nspec && row < paircnt)
  {
    size_t idx = row * ncfpitch + col;
    size_t srcrow, starow;
    size_t srcidx, staidx;
    srcrow = d_pairlist[row].srcidx;
    starow = d_pairlist[row].staidx;
    srcidx = srcrow * srcpitch + srcoffset + col;
    staidx = starow * stapitch + staoffset + col;

    // cuComplex src = d_spec[srcidx];
    // cuComplex sta_conj = make_cuComplex(d_spec[staidx].x, -d_spec[staidx].y);

    cuComplex sta = d_spec[staidx];
    cuComplex src_conj = make_cuComplex(d_spec[srcidx].x, -d_spec[srcidx].y);

    if (col == 0)
    {
      d_segncf[idx] = make_cuComplex(0, 0);
    }
    else
    {
      cuComplex mul_result = cuCmulf(src_conj, sta);
      int sign = (col % 2 == 0) ? 1 : -1;
      d_segncf[idx].x = sign * mul_result.x;
      d_segncf[idx].y = sign * mul_result.y;
    }
  }
}

// sum2dKernel is used to sum the 2D array of float, not used in the current version
__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height,
                            int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    d_finalccvec[didx] += (d_segncfvec[sidx] / nstep);
  }
}

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    cuComplex temp = d_segment_spectrum[sidx];
    temp.x /= nstep; // divide the real part by nstep
    temp.y /= nstep; // divide the imaginary part by nstep

    d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
  }
}

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  double weight = 1.0 / (width * dt);
  if (row < height && col < width)
  {
    size_t idx = row * pitch + col;
    d_segdata[idx] *= weight;
  }
}
