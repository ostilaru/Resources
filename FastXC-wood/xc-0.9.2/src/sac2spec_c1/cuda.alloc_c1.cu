#include "cuda.alloc_c1.cuh"
size_t EstimateGpuBatchC1(size_t gpu_id, int npts, int nfft_1x, int nstep, int filter_count, size_t wh_flag, size_t runabs_flag)
{
    int nfft_2x = nfft_1x * 2;

    // CuFFT parameter
    int rank = 1;
    int n[1] = {nfft_1x};
    int inembed[1] = {nfft_1x};
    int onembed[1] = {nfft_1x};
    int istride = 1;
    int idist = nfft_1x;
    int ostride = 1;
    int odist = nfft_1x;

    // CuFFT parameter for 2x zero padding data
    int rank_2x = 1;
    int n_2x[1] = {nfft_2x};
    int inembed_2x[1] = {nfft_2x};
    int onembed_2x[1] = {nfft_2x};
    int istride_2x = 1;
    int idist_2x = nfft_2x;
    int ostride_2x = 1;
    int odist_2x = nfft_2x;

    // unitgpuram setting
    size_t sac_size = npts * sizeof(float);                         // d_sacdata
    size_t spec_size = nstep * (nfft_2x + 1) * sizeof(cuComplex);   // d_spectrum
    size_t sac_seg_size = nfft_1x * sizeof(float);                  // d_segsac
    size_t spec_seg_size = nfft_1x * sizeof(cuComplex);             // d_segspec
    size_t sac_seg_2x_size = nfft_2x * sizeof(float);               // d_segsac_2x with zero padding
    size_t spec_seg_2x_size = nfft_2x * sizeof(cuComplex);          // d_segspec_2x with zero padding

    // gpuram for preprocessing rdc and rtr
    size_t pre_process_size = sizeof(double)    // d_sum
                              + sizeof(double); // d_isum

    size_t fixed_size = nfft_1x * sizeof(cuComplex) * (filter_count + 1); // fixed memory for filter response

    // calcaulate gpuram for frequency whiten and time normalization
    size_t whiten_norm_size = 0;
    if (runabs_flag)
    {                                                    // If runabs normalization is applied
        whiten_norm_size = nfft_1x * sizeof(float)       // d_weight
                           + nfft_1x * sizeof(float)     // d_segsac_tmp
                           + nfft_1x * sizeof(cuComplex) // d_segspec_tmp
                           + nfft_1x * sizeof(double);   // d_tmp
    }
    else if (!runabs_flag && wh_flag)
    {                                                 // If only frequency whiten is applied
        whiten_norm_size = nfft_1x * sizeof(float)    // d_weight
                           + nfft_1x * sizeof(float); // d_tmp
    }

    size_t unitgpuram =
        sac_size            // input sac data npts * float
        + spec_size         // output spectrum data nstep * nspec * cuComplex
        + sac_seg_size      // segment sac data nfft * float (nfft is redundant)
        + spec_seg_size     // segment spectrum data nfft * cuComplex
        + sac_seg_2x_size   // zero-padding segment spectrum data nfft * float
        + spec_seg_2x_size  // zero-padding segment spectrum data nfft * cuComplex
        + pre_process_size  // d_sum, d_isum
        + whiten_norm_size; // whiten and normalization

    size_t availram = QueryAvailGpuRam(gpu_id);
    size_t reqram = 0;
    size_t tmpram = 0;
    size_t batch = 0;
    while (true)
    {
        batch++;
        reqram = fixed_size + batch * unitgpuram;
        // cuFFT memory usage for data fft forward
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_R2C, batch, &tmpram);
        reqram += tmpram;
        // cuFFT memory usage for data fft inverse
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_C2R, batch, &tmpram);
        
        reqram += tmpram;
        
        // cuFFT memory usage for zero padding data fft forward
        cufftEstimateMany(rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x, ostride_2x,
                          odist_2x, CUFFT_R2C, batch, &tmpram);

        reqram += tmpram;

        if (reqram > availram)
        {          // Check if reqram exceeds availram
            break; // Exit the loop
        }
    }
    batch = batch > _RISTRICT_MAX_GPU_BATCH ? _RISTRICT_MAX_GPU_BATCH : batch;
    return batch;
}

void allocateCudaMemoryC1(int d_batch, int npts, int nstep_valid, int nfft_1x,
                          int do_runabs, int wh_flag,
                          float **d_timesignal,
                          cuComplex **d_spectrum,
                          float **d_segment_timesignal,
                          cuComplex **d_segment_spectrum,
                          float **d_segment_timesignal_2x,
                          cuComplex **d_segment_spectrum_2x,
                          float **d_filtered_segment_timesignal,
                          cuComplex **d_filtered_segment_spectrum,
                          cuComplex **d_filter_responses,
                          int filterCount,
                          float **d_weight, float **d_tmp,
                          double **d_sum, double **d_isum,
                          cufftHandle *planfwd, cufftHandle *planinv, cufftHandle *planfwd_2x)
{
    // Variables for input and output
    int nfft_2x = nfft_1x * 2;
    cudaMalloc((void **)d_timesignal, d_batch * npts * sizeof(float));
    cudaMalloc((void **)d_spectrum, d_batch * nstep_valid * (nfft_2x / 2 + 1) * sizeof(cuComplex));

    // Variables for processing segment data
    cudaMalloc((void **)d_segment_timesignal, d_batch * nfft_1x * sizeof(float));
    cudaMalloc((void **)d_segment_spectrum, d_batch * nfft_1x * sizeof(cuComplex));
    cudaMalloc((void **)d_segment_timesignal_2x, d_batch * nfft_2x * sizeof(float));
    cudaMalloc((void **)d_segment_spectrum_2x, d_batch * nfft_2x * sizeof(cuComplex));

    cudaMalloc((void **)d_filter_responses, filterCount * nfft_1x * sizeof(cuComplex));
    cudaMalloc((void **)d_sum, d_batch * sizeof(double));
    cudaMalloc((void **)d_isum, d_batch * sizeof(double));

    // if whiten, allocate memory for weight and tmp, same size as segment data
    if (!do_runabs && wh_flag)
    {
        cudaMalloc((void **)d_weight, d_batch * nfft_1x * sizeof(float));
        cudaMalloc((void **)d_tmp, d_batch * nfft_1x * sizeof(float));
    } // if runabs, allocate memory for filterd sac and spec
    else if (do_runabs)
    {
        cudaMalloc((void **)d_filtered_segment_timesignal, d_batch * nfft_1x * sizeof(float));
        cudaMalloc((void **)d_filtered_segment_spectrum, d_batch * nfft_1x * sizeof(cuComplex));
        cudaMalloc((void **)d_weight, d_batch * nfft_1x * sizeof(float));
        cudaMalloc((void **)d_tmp, d_batch * nfft_1x * sizeof(float));
    }

    // set up cufft plans
    int rank = 1;
    int n[1] = {nfft_1x};
    int inembed[1] = {nfft_1x};
    int onembed[1] = {nfft_1x};
    int istride = 1;
    int idist = nfft_1x;
    int ostride = 1;
    int odist = nfft_1x;

    cufftPlanMany(planfwd, rank, n, inembed, istride, idist, onembed,
                  ostride, odist, CUFFT_R2C, d_batch);
    cufftPlanMany(planinv, rank, n, inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2R, d_batch);
    
    // set up cufft plans for zero-padding
    int rank_2x = 1;
    int n_2x[1] = {nfft_2x};
    int inembed_2x[1] = {nfft_2x};
    int onembed_2x[1] = {nfft_2x};
    int istride_2x = 1;
    int idist_2x = nfft_2x;
    int ostride_2x = 1;
    int odist_2x = nfft_2x;

    cufftPlanMany(planfwd_2x, rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x,
                  ostride_2x, odist_2x, CUFFT_R2C, d_batch);
}

// Free memory for the PathNode linked list
void free_PathList(PathNode *head)
{
    PathNode *current = head;
    PathNode *next_node;

    while (current != NULL)
    {
        next_node = current->next;

        // Free memory for the string
        if (current->path != NULL)
        {
            CpuFree((void **)&current->path);
        }

        // Free memory for the struct itself
        CpuFree((void **)&current);

        current = next_node;
    }
}

void freeMemory(cufftHandle planfwd, cufftHandle planinv,
                float **d_timesignal,
                cuComplex **d_spectrum,
                float **d_segment_timesignal,
                cuComplex **d_segment_spectrum,
                float **d_segment_timesignal_2x,
                cuComplex **d_segment_spectrum_2x,
                float **d_filtered_segment_timesignal,
                cuComplex **d_filtered_segment_spectrum,
                cuComplex **d_filter_responses,
                float **d_weight, float **d_tmp,
                double **d_sum, double **d_isum,
                float **h_timesignal, complex **h_spectrum,
                InOutNodeC1 *pInOutList)
{
    cufftDestroy(planfwd);
    cufftDestroy(planinv);

    GpuFree((void **)d_timesignal);
    GpuFree((void **)d_spectrum);

    GpuFree((void **)d_segment_timesignal);
    GpuFree((void **)d_segment_spectrum);

    GpuFree((void **)d_segment_timesignal_2x);
    GpuFree((void **)d_segment_spectrum_2x);

    GpuFree((void **)d_filtered_segment_timesignal);
    GpuFree((void **)d_filtered_segment_spectrum);
    GpuFree((void **)d_filter_responses);

    GpuFree((void **)d_weight);
    GpuFree((void **)d_tmp);

    GpuFree((void **)d_sum);
    GpuFree((void **)d_isum);

    CpuFree((void **)h_timesignal);
    CpuFree((void **)h_spectrum);
    CpuFree((void **)pInOutList);
}
