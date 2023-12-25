#include "cuda.alloc_c1.cuh"

void allocateCudaMemoryC1(int d_batch, int npts, int nstep_valid, int nfft_2x,
                          int do_runabs, int wh_flag,
                          float **d_timesignal,
                          cuComplex **d_spectrum,
                          float **d_segment_timesignal,
                          cuComplex **d_segment_spectrum,
                          float **d_filtered_segment_timesignal,
                          cuComplex **d_filtered_segment_spectrum,
                          cuComplex **d_filter_responses,
                          int filterCount,
                          float **d_weight, float **d_tmp,
                          double **d_sum, double **d_isum,
                          cufftHandle *planfwd, cufftHandle *planinv)
{
    // Variables for input and output
    cudaMalloc((void **)d_timesignal, d_batch * npts * sizeof(float));
    cudaMalloc((void **)d_spectrum, d_batch * nstep_valid * (nfft_2x / 2 + 1) * sizeof(cuComplex));

    // Variables for processing segment data
    cudaMalloc((void **)d_segment_timesignal, d_batch * nfft_2x * sizeof(float));
    cudaMalloc((void **)d_segment_spectrum, d_batch * nfft_2x * sizeof(cuComplex));
    cudaMalloc((void **)d_filter_responses, filterCount * nfft_2x * sizeof(cuComplex));
    cudaMalloc((void **)d_sum, d_batch * sizeof(double));
    cudaMalloc((void **)d_isum, d_batch * sizeof(double));

    // if whiten, allocate memory for weight and tmp, same size as segment data
    if (!do_runabs && wh_flag)
    {
        cudaMalloc((void **)d_weight, d_batch * nfft_2x * sizeof(float));
        cudaMalloc((void **)d_tmp, d_batch * nfft_2x * sizeof(float));
    } // if runabs, allocate memory for filterd sac and spec
    else if (do_runabs)
    {
        cudaMalloc((void **)d_filtered_segment_timesignal, d_batch * nfft_2x * sizeof(float));
        cudaMalloc((void **)d_filtered_segment_spectrum, d_batch * nfft_2x * sizeof(cuComplex));
        cudaMalloc((void **)d_weight, d_batch * nfft_2x * sizeof(float));
        cudaMalloc((void **)d_tmp, d_batch * nfft_2x * sizeof(float));
    }

    // set up cufft plans
    int rank = 1;
    int n[1] = {nfft_2x};
    int inembed[1] = {nfft_2x};
    int onembed[1] = {nfft_2x};
    int istride = 1;
    int idist = nfft_2x;
    int ostride = 1;
    int odist = nfft_2x;

    cufftPlanMany(planfwd, 1, n, inembed, istride, idist, onembed,
                  ostride, odist, CUFFT_R2C, d_batch);
    cufftPlanMany(planinv, rank, n, inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2R, d_batch);
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
