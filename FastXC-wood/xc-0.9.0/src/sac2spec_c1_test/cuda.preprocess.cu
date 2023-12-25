#include "cuda.preprocess.cuh"

// pre-processing for sacdat: isnan, demean, detrend
void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int npts,
                int pitch, size_t proccnt, int taper_percentage)
{
    // check the nan/inf value of [d_sacdata]
    size_t width = npts;
    size_t height = proccnt;
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    dim3 dimgrd2, dimblk2;
    dimblk2.x = BLOCKMAX;
    dimblk2.y = 1;
    dimgrd2.x = 1;
    dimgrd2.y = height;

    isnan2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height);

    // demean. First calculate the mean value of each trace

    size_t dpitch = 1;
    size_t spitch = pitch;
    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    DimCompute(&dimgrd, &dimblk, width, height);
    rdc2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height,
                                    d_sum);

    // detrend. First calculate d_sum and d_isum

    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    isumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                              dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_isum, dpitch, d_sacdata, spitch, width, height);

    rtr2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height,
                                    d_sum, d_isum);

    // taper, taper percentage set in config.h
    timetaper2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, taper_percentage);
}
