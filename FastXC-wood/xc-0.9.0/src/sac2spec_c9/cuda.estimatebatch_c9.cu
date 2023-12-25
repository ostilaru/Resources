#include "cuda.estimatebatch_c9.cuh"

size_t EstimateGpuBatchC9(int gpu_id, int npts, int nfft_2x, int nstep,
                          size_t wh_flag, size_t runabs_flag)
{
    // CuFFT parameter
    int rank = 1;
    int n[1] = {nfft_2x};
    int inembed[1] = {nfft_2x};
    int onembed[1] = {nfft_2x};
    int istride = 1;
    int idist = nfft_2x;
    int ostride = 1;
    int odist = nfft_2x;

    // unitgpuram setting
    size_t sac_size = npts * sizeof(float);                       // d_sacdata
    size_t spec_size = nstep * (nfft_2x + 1) * sizeof(cuComplex); // d_specdata
    size_t sac_seg_size = nfft_2x * sizeof(float);                // d_segsac
    size_t spec_seg_size = nfft_2x * sizeof(cuComplex);           // d_segspec

    // gpuram for preprocessing rdc and rtr
    size_t pre_process_size = sizeof(double)    // d_sum
                              + sizeof(double); // d_isum

    // calcaulate gpuram for frequency whiten and time normalization
    size_t whiten_norm_size = 0;
    if (runabs_flag)
    {                                                    // If runabs normalization is applied
        whiten_norm_size = nfft_2x * sizeof(float)       // d_weight
                           + nfft_2x * sizeof(float)     // d_segsac_tmp
                           + nfft_2x * sizeof(cuComplex) // d_segspec_tmp
                           + nfft_2x * sizeof(double);   // d_tmp
    }
    else if (!runabs_flag && wh_flag)
    {                                                 // If only frequency whiten is applied
        whiten_norm_size = nfft_2x * sizeof(float)    // d_weight
                           + nfft_2x * sizeof(float); // d_tmp
    }

    // Use 3* for three (Nine) compoents condition
    size_t unitgpuram =
        3 * sac_size           // input sac data npts * float
        + 3 * spec_size        // output spectrum data nstep * nspec * cuComplex
        + 3 * sac_seg_size     // segment sac data nfft * float (nfft is redundant)
        + 3 * spec_seg_size    // segment spectrum data nfft * cuComplex
        + 3 * pre_process_size // d_sum, d_isum
        + whiten_norm_size;    // whiten and normalization

    size_t availram = QueryAvailGpuRam(gpu_id);
    size_t reqram = 0;
    size_t tmpram = 0;
    size_t batch = 0;
    while (true)
    {
        batch++;
        reqram = batch * unitgpuram;
        // cuFFT memory usage for data fft forward
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_R2C, batch, &tmpram);
        reqram += tmpram;
        // cuFFT memory usage for data fft inverse
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_C2R, batch, &tmpram);
        reqram += tmpram;

        if (reqram > availram)
        {          // Check if reqram exceeds availram
            break; // Exit the loop
        }
    }
    batch = batch > _RISTRICT_MAX_GPU_BATCH ? _RISTRICT_MAX_GPU_BATCH : batch;
    return batch;
}