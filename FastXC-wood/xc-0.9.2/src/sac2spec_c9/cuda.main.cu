/* last updated by wangjx@20240103 */

#include "cuda.alloc_c9.cuh"
#include "cuda.processing_c9.cuh"
#include "cuda.util.cuh"

extern "C"
{
#include "design_filter_response.h"
#include "arguproc.h"
#include "in_out_node_c9.h"
#include "path_node.h"
#include "cal_nseg.h"
#include "read_filelist.h"
#include "par_read_sac_c9.h"
#include "par_write_spec_c9.h"
#include "sac.h"
#include "util.h"
}

void find_whiten_flag(int whitenType, int normalizeType, int *wh_before,
                      int *wh_after, int *do_runabs, int *do_onebit)
{
    // Determine bandwhiten flags
    switch (whitenType)
    {
    case 0:
        *wh_before = 0;
        *wh_after = 0;
        break;
    case 1:
        *wh_before = 1;
        *wh_after = 0;
        break;
    case 2:
        *wh_before = 0;
        *wh_after = 1;
        break;
    case 3:
        *wh_before = 1;
        *wh_after = 1;
        break;
    default:
        // Invalid value for bandwhiten
        printf("Invalid value for bandwhiten\n");
        // Handle error or set default values
        break;
    }

    // Determine normalization flags
    switch (normalizeType)
    {
    case 0:
        *do_runabs = 0;
        *do_onebit = 0;
        break;
    case 1:
        *do_runabs = 1;
        *do_onebit = 0;
        break;
    case 2:
        *do_runabs = 0;
        *do_onebit = 1;
        break;
    default:
        // Invalid value for normalization
        printf("Invalid value for normalization\n");
        // Handle error or set default values
        break;
    }
}


int main(int argc, char **argv)
{
    // Parsing arguments
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);

    // read in sac path list file
    PathNode *pInFileList_1 = readPathList(argument.sacin_lst_1);
    PathNode *pInFileList_2 = readPathList(argument.sacin_lst_2);
    PathNode *pInFileList_3 = readPathList(argument.sacin_lst_3);

    // read in spec output path list file
    PathNode *pOutFileList_1 = readPathList(argument.specout_lst_1);
    PathNode *pOutFileList_2 = readPathList(argument.specout_lst_2);
    PathNode *pOutFileList_3 = readPathList(argument.specout_lst_3);

    // create output_dir for .segspec files
    createDirectories(pOutFileList_1);
    createDirectories(pOutFileList_2);
    createDirectories(pOutFileList_3);

    // Turn file chain input list chain into array
    FilePathArray InPaths_1 = PathList2Array(pInFileList_1);
    FilePathArray InPaths_2 = PathList2Array(pInFileList_2);
    FilePathArray InPaths_3 = PathList2Array(pInFileList_3);

    FilePathArray OutPaths_1 = PathList2Array(pOutFileList_1);
    FilePathArray OutPaths_2 = PathList2Array(pOutFileList_2);
    FilePathArray OutPaths_3 = PathList2Array(pOutFileList_3);

    // Parse whiten and normalization parameters
    int wh_before = 0, wh_after = 0, do_runabs = 0, do_onebit = 0;
    find_whiten_flag(argument.whitenType, argument.normalizeType, &wh_before, &wh_after, &do_runabs, &do_onebit);

    // Parse frequcny band
    float freq_low_limit = argument.freq_low_limit;
    float freq_high_limit = argument.freq_high_limit;

    // count the number of input files
    size_t nValid = InPaths_1.count;

    // set the gpu_id
    size_t gpu_id = argument.gpu_id;
    cudaSetDevice(gpu_id);

    // read npts and delta form the file file of pInFileList
    SACHEAD sachd;
    if (read_sachead(pInFileList_1->path, &sachd) != 0)
    {
        fprintf(stderr, "ERROR reading first SACHEAD\n");
        exit(EXIT_FAILURE);
    }
    int npts = sachd.npts;
    float delta = sachd.delta;

    // calculate the nseg
    int nseg = cal_nseg(argument.seglen, npts, delta);
    int nstep = npts / nseg;

    // create nfft_2x for zero padding data
    int nfft_2x = nseg * 2;
    int nfft_1x = nseg;

    int nspec_2x = nfft_2x / 2 + 1;
    int nspec_1x = nfft_1x / 2 + 1;

    float df_1x = 1.0 / (nfft_1x * delta);
    float df_2x = 1.0 / (nfft_2x * delta);

    // Parse the option skip_step
    int skip_step = argument.skip_step;
    int nstep_valid = nstep;
    if (skip_step != -1 && skip_step < nstep)
    {
        nstep_valid = nstep - 1;
        printf("We will skip [no. %d] step\n", skip_step);
    }

    // read in filter file
    int filter_count = 0;
    ButterworthFilter *filter = readButterworthFilters(argument.filter_file, &filter_count);
    FilterResp *myResp = processButterworthFilters(filter, filter_count, df_1x, nfft_1x);

    // set smooth npts for whitenning, modified from yao's code 0.02/df
    int nsmooth = 0.5 * int(0.02 * nfft_1x * delta);
    nsmooth = (nsmooth > 11) ? nsmooth : 11;

    // ********* Calculate CPU memory *********** //
    float *h_timesignal_1 = NULL;
    float *h_timesignal_2 = NULL;
    float *h_timesignal_3 = NULL;

    complex *h_spectrum_1 = NULL;
    complex *h_spectrum_2 = NULL;
    complex *h_spectrum_3 = NULL;

    InOutNodeC9 *pInOutList = NULL;

    size_t unit_timesignal_size = 3 * npts * sizeof(float);                   // input sac data
    size_t unit_spectrum_size = 3 * nstep_valid * nspec_2x * sizeof(complex); // output total spectrum
    size_t unit_InOutNode_size = sizeof(InOutNodeC9);                         // contain head/path/data
    size_t unit_thread_write_size = sizeof(thread_info_write);                // contain thread_info_write
    size_t unit_thread_read_size = sizeof(thread_info_read);                  // contain thread_info_read

    size_t unitCpuRam = unit_timesignal_size +
                        unit_spectrum_size +
                        unit_InOutNode_size +
                        unit_thread_write_size +
                        unit_thread_read_size;

    size_t h_batch = EstimateCpuBatch(unitCpuRam, argument.thread_num);
    h_batch = (h_batch > nValid) ? nValid : h_batch;

    // Allocate memory for GPU
    size_t wh_flag = wh_after || wh_before;
    size_t d_batch = EstimateGpuBatchC9(gpu_id, npts, nfft_1x, nstep, filter_count,
                                        wh_flag, do_runabs);

    d_batch = (d_batch > h_batch) ? h_batch : d_batch;
    h_batch = d_batch;

    printf("[GPU no.%zu]: cpu_batch = %ld\n", gpu_id, h_batch);
    printf("[GPU no.%zu]: gpu_batch = %ld\n", gpu_id, d_batch);

    // Allocate CPU memory
    CpuMalloc((void **)&pInOutList, h_batch * unit_InOutNode_size);

    CpuMalloc((void **)&h_timesignal_1, h_batch * unit_timesignal_size);
    CpuMalloc((void **)&h_spectrum_1, h_batch * unit_spectrum_size);

    CpuMalloc((void **)&h_timesignal_2, h_batch * unit_timesignal_size);
    CpuMalloc((void **)&h_spectrum_2, h_batch * unit_spectrum_size);

    CpuMalloc((void **)&h_timesignal_3, h_batch * unit_timesignal_size);
    CpuMalloc((void **)&h_spectrum_3, h_batch * unit_spectrum_size);

    // Initializing the memory for InOutNode
    for (size_t i = 0; i < h_batch; i++)
    {
        size_t sacpathSize = MAXPATH * sizeof(char);
        size_t specpathSize = MAXPATH * sizeof(char);
        size_t spechdSize = sizeof(SEGSPEC);
        size_t sachdSize = sizeof(SACHEAD);
        CpuMalloc((void **)&(pInOutList[i].sacpath_1), sacpathSize);
        CpuMalloc((void **)&(pInOutList[i].sacpath_2), sacpathSize);
        CpuMalloc((void **)&(pInOutList[i].sacpath_3), sacpathSize);

        CpuMalloc((void **)&(pInOutList[i].specpath_1), specpathSize);
        CpuMalloc((void **)&(pInOutList[i].specpath_2), specpathSize);
        CpuMalloc((void **)&(pInOutList[i].specpath_3), specpathSize);

        CpuMalloc((void **)&(pInOutList[i].sac_hd), sachdSize);
        CpuMalloc((void **)&(pInOutList[i].segspec_hd), spechdSize);

        pInOutList[i].timesignal_1 = h_timesignal_1 + i * npts;
        pInOutList[i].spectrum_1 = h_spectrum_1 + i * nstep_valid * nspec_2x;

        pInOutList[i].timesignal_2 = h_timesignal_2 + i * npts;
        pInOutList[i].spectrum_2 = h_spectrum_2 + i * nstep_valid * nspec_2x;

        pInOutList[i].timesignal_3 = h_timesignal_3 + i * npts;
        pInOutList[i].spectrum_3 = h_spectrum_3 + i * nstep_valid * nspec_2x;

        pInOutList[i].nspec = nspec_2x;
        pInOutList[i].nstep = nstep_valid;
        pInOutList[i].df = df_2x;
        pInOutList[i].dt = delta;
    }

    // ********** Allocate memory for GPU ************ //

    // Initialize the memory for GPU
    float *d_timesignal_1 = NULL;
    float *d_timesignal_2 = NULL;
    float *d_timesignal_3 = NULL;

    cuComplex *d_spectrum_1 = NULL;
    cuComplex *d_spectrum_2 = NULL;
    cuComplex *d_spectrum_3 = NULL;

    float *d_segment_timesignal_1 = NULL;
    float *d_segment_timesignal_2 = NULL;
    float *d_segment_timesignal_3 = NULL;

    float *d_segment_timesignal_1_2x = NULL;
    float *d_segment_timesignal_2_2x = NULL;
    float *d_segment_timesignal_3_2x = NULL;

    cuComplex *d_segment_spectrum_1 = NULL;
    cuComplex *d_segment_spectrum_2 = NULL;
    cuComplex *d_segment_spectrum_3 = NULL;

    cuComplex *d_segment_spectrum_1_2x = NULL;
    cuComplex *d_segment_spectrum_2_2x = NULL;
    cuComplex *d_segment_spectrum_3_2x = NULL;

    float *d_filtered_segment_timesignal_1 = NULL;
    float *d_filtered_segment_timesignal_2 = NULL;
    float *d_filtered_segment_timesignal_3 = NULL;

    cuComplex *d_filtered_segment_spectrum_1 = NULL;
    cuComplex *d_filtered_segment_spectrum_2 = NULL;
    cuComplex *d_filtered_segment_spectrum_3 = NULL;

    cuComplex *d_filter_responses = NULL;

    float *d_weight_1 = NULL; // weight for each segment spectrum data
    float *d_weight_2 = NULL;
    float *d_weight_3 = NULL;

    float *d_tmp = NULL;   // used in npsmooth
    double *d_sum = NULL;  // used in rtr and rdc
    double *d_isum = NULL; // used in rtr and rdc

    cufftHandle planfwd; // forward fft
    cufftHandle planinv; // inverse fft

    cufftHandle planfwd_2x; // forward fft, for 2x zero-padding series
    // Allocating memory in GPU Device
    allocateCudaMemoryC9(d_batch, npts, nstep_valid, nfft_1x,
                         do_runabs, wh_flag,
                         &d_timesignal_1, &d_spectrum_1,
                         &d_timesignal_2, &d_spectrum_2,
                         &d_timesignal_3, &d_spectrum_3,
                         &d_segment_timesignal_1, &d_segment_spectrum_1,
                         &d_segment_timesignal_2, &d_segment_spectrum_2,
                         &d_segment_timesignal_3, &d_segment_spectrum_3,
                         &d_segment_timesignal_1_2x, &d_segment_spectrum_1_2x,
                         &d_segment_timesignal_2_2x, &d_segment_spectrum_2_2x,
                         &d_segment_timesignal_3_2x, &d_segment_spectrum_3_2x,
                         &d_filtered_segment_timesignal_1, &d_filtered_segment_spectrum_1,
                         &d_filtered_segment_timesignal_2, &d_filtered_segment_spectrum_2,
                         &d_filtered_segment_timesignal_3, &d_filtered_segment_spectrum_3,
                         &d_filter_responses,
                         filter_count,
                         &d_weight_1, &d_weight_2, &d_weight_3,
                         &d_tmp, &d_sum, &d_isum,
                         &planfwd, &planinv, &planfwd_2x);

    // ********** MAIN PROCESSING PART ********** /

    // copy each filter response to GPU and store freq_low array
    float freq_lows[filter_count];
    size_t offset;
    for (int i = 0; i < filter_count; i++)
    {
        freq_lows[i] = myResp[i].freq_low;
        // Calculate the offset of each filter response
        offset = i * nfft_1x;
        CUDACHECK(cudaMemcpy2D(d_filter_responses + offset, nfft_1x * sizeof(cuComplex), myResp[i].response, nfft_1x * sizeof(cuComplex),
                               nfft_1x * sizeof(cuComplex), 1, cudaMemcpyHostToDevice));
    }

    for (size_t h_finishcnt = 0; h_finishcnt < nValid; h_finishcnt += h_batch)
    {
        size_t h_proccnt = (h_finishcnt + h_batch > nValid) ? nValid - h_finishcnt : h_batch;

        // Setup Input Output Chain
        for (size_t i = h_finishcnt, j = 0; i < h_finishcnt + h_proccnt; i++, j++)
        {
            pInOutList[j].sacpath_1 = InPaths_1.paths[i];
            pInOutList[j].specpath_1 = OutPaths_1.paths[i];

            pInOutList[j].sacpath_2 = InPaths_2.paths[i];
            pInOutList[j].specpath_2 = OutPaths_2.paths[i];

            pInOutList[j].sacpath_3 = InPaths_3.paths[i];
            pInOutList[j].specpath_3 = OutPaths_3.paths[i];
        }

        ThreadPoolRead *read_pool = create_threadpool_read(MAX_THREADS);
        ThreadPoolWrite *write_pool = create_threadpool_write(MAX_THREADS);

        // Set [h_timesignal_*] and [h_spectrum_*] to zero
        memset(h_timesignal_1, 0, h_proccnt * npts * sizeof(float));
        memset(h_spectrum_1, 0, h_proccnt * nstep_valid * nspec_2x * sizeof(complex));

        memset(h_timesignal_2, 0, h_proccnt * npts * sizeof(float));
        memset(h_spectrum_2, 0, h_proccnt * nstep_valid * nspec_2x * sizeof(complex));

        memset(h_timesignal_3, 0, h_proccnt * npts * sizeof(float));
        memset(h_spectrum_3, 0, h_proccnt * nstep_valid * nspec_2x * sizeof(complex));

        parallel_read_sac_c9(read_pool, h_proccnt, pInOutList, MAX_THREADS);
        // Launch GPU Processing Part

        dim3 dimgrd_1x, dimblk_1x, dimgrd_2x, dimblk_2x;
        for (size_t d_finishcnt = 0; d_finishcnt < h_proccnt; d_finishcnt += d_batch)
        {
            size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? h_proccnt - d_finishcnt : d_batch;

            CUDACHECK(cudaMemset(d_timesignal_1, 0, d_proccnt * npts * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum_1, 0, d_proccnt * nstep_valid * nspec_2x * sizeof(cuComplex)));

            CUDACHECK(cudaMemset(d_timesignal_2, 0, d_proccnt * npts * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum_2, 0, d_proccnt * nstep_valid * nspec_2x * sizeof(cuComplex)));

            CUDACHECK(cudaMemset(d_timesignal_3, 0, d_proccnt * npts * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum_3, 0, d_proccnt * nstep_valid * nspec_2x * sizeof(cuComplex)));

            // Read SAC Data
            CUDACHECK(cudaMemcpy2D(d_timesignal_1, npts * sizeof(float),
                                   h_timesignal_1 + d_finishcnt * npts, npts * sizeof(float),
                                   npts * sizeof(float), d_proccnt, cudaMemcpyHostToDevice));

            CUDACHECK(cudaMemcpy2D(d_timesignal_2, npts * sizeof(float),
                                   h_timesignal_2 + d_finishcnt * npts, npts * sizeof(float),
                                   npts * sizeof(float), d_proccnt, cudaMemcpyHostToDevice));

            CUDACHECK(cudaMemcpy2D(d_timesignal_3, npts * sizeof(float),
                                   h_timesignal_3 + d_finishcnt * npts, npts * sizeof(float),
                                   npts * sizeof(float), d_proccnt, cudaMemcpyHostToDevice));

            for (int stepidx = 0, done_step_flag = 0; stepidx < nstep; stepidx++)
            {
                // skip any step?
                if (skip_step == stepidx)
                    continue;

                // clean d_segment_timesignal_* and d_segment_spectrum_*
                CUDACHECK(cudaMemset(d_segment_timesignal_1, 0, d_proccnt * nfft_1x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_1, 0, d_proccnt * nfft_1x * sizeof(cuComplex)));

                CUDACHECK(cudaMemset(d_segment_timesignal_2, 0, d_proccnt * nfft_1x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_2, 0, d_proccnt * nfft_1x * sizeof(cuComplex)));

                CUDACHECK(cudaMemset(d_segment_timesignal_3, 0, d_proccnt * nfft_1x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_3, 0, d_proccnt * nfft_1x * sizeof(cuComplex)));

                CUDACHECK(cudaMemset(d_segment_timesignal_1_2x, 0, d_proccnt * nfft_2x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_1_2x, 0, d_proccnt * nfft_2x * sizeof(cuComplex)));

                CUDACHECK(cudaMemset(d_segment_timesignal_2_2x, 0, d_proccnt * nfft_2x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_2_2x, 0, d_proccnt * nfft_2x * sizeof(cuComplex)));

                CUDACHECK(cudaMemset(d_segment_timesignal_3_2x, 0, d_proccnt * nfft_2x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum_3_2x, 0, d_proccnt * nfft_2x * sizeof(cuComplex)));

                // copy d_timesignal_* to d_segment_timesignal_*
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_1, nfft_1x * sizeof(float),
                                       d_timesignal_1 + stepidx * nseg, npts * sizeof(float),
                                       nseg * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_2, nfft_1x * sizeof(float),
                                       d_timesignal_2 + stepidx * nseg, npts * sizeof(float),
                                       nseg * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_3, nfft_1x * sizeof(float),
                                       d_timesignal_3 + stepidx * nseg, npts * sizeof(float),
                                       nseg * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));

                // pre-processing of d_segment_timesignal_*
                preprocess(d_segment_timesignal_1, d_sum, d_isum, nseg, nfft_1x, d_proccnt, TAPER_RATIO);
                preprocess(d_segment_timesignal_2, d_sum, d_isum, nseg, nfft_1x, d_proccnt, TAPER_RATIO);
                preprocess(d_segment_timesignal_3, d_sum, d_isum, nseg, nfft_1x, d_proccnt, TAPER_RATIO);

                DimCompute(&dimgrd_1x, &dimblk_1x, nfft_1x, d_proccnt);
                if (wh_before || do_runabs)
                {
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_1, (cufftComplex *)d_segment_spectrum_1));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_1, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_2, (cufftComplex *)d_segment_spectrum_2));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_2, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_3, (cufftComplex *)d_segment_spectrum_3));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_3, nfft_1x, nfft_1x, d_proccnt, delta);

                    if (wh_before)
                    {
                        freqWhiten_c9(d_segment_spectrum_1, d_segment_spectrum_2, d_segment_spectrum_3,
                                      d_weight_1, d_weight_2, d_weight_3,
                                      d_tmp, nfft_1x, nspec_1x, d_proccnt, nsmooth, df_1x, freq_low_limit, freq_high_limit);
                    }
                    if (do_onebit)
                    {

                        CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_1, (cufftReal *)d_segment_timesignal_1));
                        InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_1, nfft_1x, nfft_1x, d_proccnt, delta);

                        CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_2, (cufftReal *)d_segment_timesignal_2));
                        InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_2, nfft_1x, nfft_1x, d_proccnt, delta);

                        CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_3, (cufftReal *)d_segment_timesignal_3));
                        InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_3, nfft_1x, nfft_1x, d_proccnt, delta);
                    }
                }
                if (do_runabs)
                {
                    runabs_c9(d_segment_timesignal_1,
                              d_segment_timesignal_2,
                              d_segment_timesignal_3,
                              d_segment_spectrum_1,
                              d_segment_spectrum_2,
                              d_segment_spectrum_3,
                              d_filtered_segment_timesignal_1,
                              d_filtered_segment_timesignal_2,
                              d_filtered_segment_timesignal_3,
                              d_filtered_segment_spectrum_1,
                              d_filtered_segment_spectrum_2,
                              d_filtered_segment_spectrum_3,
                              d_weight_1, d_weight_2, d_weight_3,
                              d_tmp,
                              &planinv,
                              d_filter_responses, freq_lows,
                              filter_count, delta, d_proccnt, nfft_1x, MAXVAL);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_1, (cufftComplex *)d_segment_spectrum_1));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_1, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_2, (cufftComplex *)d_segment_spectrum_2));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_2, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_3, (cufftComplex *)d_segment_spectrum_3));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_3, nfft_1x, nfft_1x, d_proccnt, delta);
                }

                if (do_onebit)
                {
                    onebit2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_1, nfft_1x, nfft_1x, d_proccnt);
                    onebit2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_2, nfft_1x, nfft_1x, d_proccnt);
                    onebit2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_3, nfft_1x, nfft_1x, d_proccnt);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_1, (cufftComplex *)d_segment_spectrum_1));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_1, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_2, (cufftComplex *)d_segment_spectrum_2));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_2, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_3, (cufftComplex *)d_segment_spectrum_3));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_3, nfft_1x, nfft_1x, d_proccnt, delta);
                }

                if (!do_runabs && !do_onebit && !wh_before)
                {
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_1, (cufftComplex *)d_segment_spectrum_1));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_1, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_2, (cufftComplex *)d_segment_spectrum_2));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_2, nfft_1x, nfft_1x, d_proccnt, delta);

                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal_3, (cufftComplex *)d_segment_spectrum_3));
                    FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_spectrum_3, nfft_1x, nfft_1x, d_proccnt, delta);
                }

                if (wh_after)
                {
                    freqWhiten_c9(d_segment_spectrum_1, d_segment_spectrum_2, d_segment_spectrum_3,
                                  d_weight_1, d_weight_2, d_weight_3,
                                  d_tmp, nfft_1x, nspec_1x, d_proccnt, nsmooth, df_1x,
                                  freq_low_limit, freq_high_limit);
                }

                // Zero padding
                
                // Zero padding STEP 1: transform freqeuncy domain data to time domain
                CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_1, (cufftReal *)d_segment_timesignal_1));
                CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_2, (cufftReal *)d_segment_timesignal_2));
                CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum_3, (cufftReal *)d_segment_timesignal_3));
                InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_1, nfft_1x, nfft_1x, d_proccnt, delta);
                InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_2, nfft_1x, nfft_1x, d_proccnt, delta);
                InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_segment_timesignal_3, nfft_1x, nfft_1x, d_proccnt, delta);

                // Zero padding STEP 2: zero padding
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_1_2x, nfft_2x * sizeof(float),
                                       d_segment_timesignal_1, nfft_1x * sizeof(float),
                                       nfft_1x * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_2_2x, nfft_2x * sizeof(float),
                                       d_segment_timesignal_2, nfft_1x * sizeof(float),
                                       nfft_1x * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal_3_2x, nfft_2x * sizeof(float),
                                       d_segment_timesignal_3, nfft_1x * sizeof(float),
                                       nfft_1x * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));

                // Zero padding STEP 3: IFFT
                DimCompute(&dimgrd_2x, &dimblk_2x, nfft_2x, d_proccnt);
                CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_segment_timesignal_1_2x, (cufftComplex *)d_segment_spectrum_1_2x));
                FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_segment_spectrum_1_2x, nfft_2x, nfft_2x, d_proccnt, delta);

                CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_segment_timesignal_2_2x, (cufftComplex *)d_segment_spectrum_2_2x));
                FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_segment_spectrum_2_2x, nfft_2x, nfft_2x, d_proccnt, delta);

                CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_segment_timesignal_3_2x, (cufftComplex *)d_segment_spectrum_3_2x));
                FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_segment_spectrum_3_2x, nfft_2x, nfft_2x, d_proccnt, delta);

                // only the first [nfft_2x/2+1] points are copied
                CUDACHECK(cudaMemcpy2D(d_spectrum_1 + done_step_flag * nspec_2x, nstep_valid * nspec_2x * sizeof(cuComplex),
                                       d_segment_spectrum_1_2x, nfft_2x * sizeof(cuComplex),
                                       nspec_2x * sizeof(cuComplex), d_proccnt, cudaMemcpyDeviceToDevice));

                CUDACHECK(cudaMemcpy2D(d_spectrum_2 + done_step_flag * nspec_2x, nstep_valid * nspec_2x * sizeof(cuComplex),
                                       d_segment_spectrum_2_2x, nfft_2x * sizeof(cuComplex),
                                       nspec_2x * sizeof(cuComplex), d_proccnt, cudaMemcpyDeviceToDevice));

                CUDACHECK(cudaMemcpy2D(d_spectrum_3 + done_step_flag * nspec_2x, nstep_valid * nspec_2x * sizeof(cuComplex),
                                       d_segment_spectrum_3_2x, nfft_2x * sizeof(cuComplex),
                                       nspec_2x * sizeof(cuComplex), d_proccnt, cudaMemcpyDeviceToDevice));
                done_step_flag++;
            } // End loop of each segment
            CUDACHECK(cudaMemcpy2D(h_spectrum_1 + d_finishcnt * nstep_valid * nspec_2x,
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_spectrum_1,
                                   nstep_valid * nspec_2x * sizeof(cuComplex),
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_proccnt,
                                   cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy2D(h_spectrum_2 + d_finishcnt * nstep_valid * nspec_2x,
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_spectrum_2,
                                   nstep_valid * nspec_2x * sizeof(cuComplex),
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_proccnt,
                                   cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy2D(h_spectrum_3 + d_finishcnt * nstep_valid * nspec_2x,
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_spectrum_3,
                                   nstep_valid * nspec_2x * sizeof(cuComplex),
                                   nstep_valid * nspec_2x * sizeof(cuComplex), d_proccnt,
                                   cudaMemcpyDeviceToHost));

        } // Quit GPU processing
        // Writing the output spectra
        parallel_write_spec_c9(write_pool, h_proccnt, pInOutList, MAX_THREADS);
        destroy_threadpool_write(write_pool);
        destroy_threadpool_read(read_pool);
    }
    // free memory
    freeMemory(planfwd, planinv, planfwd_2x,
               &d_timesignal_1, &d_spectrum_1,
               &d_timesignal_2, &d_spectrum_2,
               &d_timesignal_3, &d_spectrum_3,
               &d_segment_timesignal_1, &d_segment_spectrum_1,
               &d_segment_timesignal_2, &d_segment_spectrum_2,
               &d_segment_timesignal_3, &d_segment_spectrum_3,
               &d_segment_timesignal_1_2x, &d_segment_spectrum_1_2x,
               &d_segment_timesignal_2_2x, &d_segment_spectrum_2_2x,
               &d_segment_timesignal_3_2x, &d_segment_spectrum_3_2x,
               &d_filtered_segment_timesignal_1, &d_filtered_segment_spectrum_1,
               &d_filtered_segment_timesignal_2, &d_filtered_segment_spectrum_2,
               &d_filtered_segment_timesignal_3, &d_filtered_segment_spectrum_3,
               &d_filter_responses,
               &d_weight_1, &d_weight_2, &d_weight_3,
               &d_sum, &d_isum, &d_tmp,
               &h_timesignal_1, &h_spectrum_1,
               &h_timesignal_2, &h_spectrum_2,
               &h_timesignal_3, &h_spectrum_3,
               pInOutList);
}
