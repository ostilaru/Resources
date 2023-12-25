// Last updated by wang jingxi @20230605
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cuda.util.cuh"
#include "segspec.h"
#include "complex.h"
#include "cuda.alloc_c1.cuh"
#include "config.h"
#include "cuda.preprocess.cuh"
#include "cuda.whiten_c1.cuh"
#include "cuda.runabs_c1.cuh"
#include "cuda.estimatebatch_c1.cuh"
#include "cuda.onebit.cuh"
#include "path_node.h"

extern "C"
{
#include "design_filter_response.h"
#include "sac.h"
#include "arguproc.h"
#include "in_out_node_c1.h"
#include "cal_nseg.h"
#include "util.h"
#include "find_whiten_flag.h"
#include "read_filelist.h"
#include "par_read_sac_c1.h"
#include "par_write_spec_c1.h"
}

int main(int argc, char **argv)
{
    // Parsing argument
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);

    // read in SAC path list file
    PathNode *pInFileList = readPathList(argument.sacin_lst);
    PathNode *pOutFileList = readPathList(argument.specout_lst);
    createDirectories(pOutFileList);

    // Turn file list into array
    FilePathArray InPaths = PathList2Array(pInFileList);
    FilePathArray OutPaths = PathList2Array(pOutFileList);

    // Parse whiten and normalization option
    int wh_before = 0, wh_after = 0, do_runabs = 0, do_onebit = 0;
    find_whiten_flag(argument.whitenType, argument.normalizeType, &wh_before,
                     &wh_after, &do_runabs, &do_onebit);

    // Parse frequency bands
    float freq_low_limit = argument.freq_low_limit;
    float freq_high_limit = argument.freq_high_limit;

    // count the number of input files
    size_t nValid = InPaths.count;

    // set the gpu_id
    int gpu_id = argument.gpu_id;
    cudaSetDevice(gpu_id);

    // read npts and delta form the file file of pInFileList
    SACHEAD sachd;
    if (read_sachead(InPaths.paths[0], &sachd) != 0)
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
    int nspec = nfft_2x / 2 + 1;

    float df = 1.0 / (nfft_2x * delta);

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
    printButterworthFilters(filter, filter_count);
    FilterResp *myResp = processButterworthFilters(filter, filter_count, df, nspec);

    // set smooth npts for whitenning, modified from yao's code 0.02/df
    int nsmooth = int(0.02 * nfft_2x * delta);
    nsmooth = (nsmooth > 11) ? nsmooth : 11;

    // Calculate CPU memory and estimate batch size
    float *h_timesignal = NULL;
    complex *h_spectrum = NULL;
    InOutNodeC1 *pInOutList = NULL;

    size_t unit_timesignal_size = npts * sizeof(float);                // input sac data
    size_t unit_spectrum_size = nstep_valid * nspec * sizeof(complex); // output total spectrum
    size_t unit_InOutNode_size = sizeof(InOutNodeC1);                  // contain head/path/data
    size_t unit_thread_write_size = sizeof(thread_info_write);         // contain thread_info_write
    size_t unit_thread_read_size = sizeof(thread_info_read);           // contain thread_info_read

    size_t unitCpuRam = unit_timesignal_size +
                        unit_spectrum_size +
                        unit_InOutNode_size +
                        unit_thread_write_size +
                        unit_thread_read_size;

    size_t h_batch = EstimateCpuBatch(unitCpuRam, argument.thread_num);
    h_batch = (h_batch > nValid) ? nValid : h_batch;
    // Allocate memory for GPU
    size_t wh_flag = wh_after || wh_before;
    size_t d_batch = EstimateGpuBatchC1(gpu_id, npts, nfft_2x, nstep, filter_count,

                                        wh_flag, do_runabs);
    d_batch = (d_batch > h_batch) ? h_batch : d_batch;
    if (d_batch < h_batch)
    {
        h_batch = d_batch;
    }
    else
    {
        d_batch = h_batch;
    }
    printf("[GPU no.%d]: gpubatch = %ld\n", gpu_id, d_batch);
    printf("[GPU no.%d]: cpubatch = %ld\n", gpu_id, h_batch);

    // Allocate CPU memory
    CpuMalloc((void **)&h_timesignal, h_batch * unit_timesignal_size);
    CpuMalloc((void **)&h_spectrum, h_batch * unit_spectrum_size);
    CpuMalloc((void **)&pInOutList, h_batch * unit_InOutNode_size);

    // Initializing the memory for InOutNode
    for (size_t i = 0; i < h_batch; i++)
    {
        size_t sacpathSize = MAXPATH * sizeof(char);
        size_t specpathSize = MAXPATH * sizeof(char);
        size_t spechdSize = sizeof(SEGSPEC);
        size_t sachdSize = sizeof(SACHEAD);
        CpuMalloc((void **)&(pInOutList[i].sacpath), sacpathSize);
        CpuMalloc((void **)&(pInOutList[i].specpath), specpathSize);
        CpuMalloc((void **)&(pInOutList[i].segspec_hd), spechdSize);
        CpuMalloc((void **)&(pInOutList[i].sac_hd), sachdSize);
        pInOutList[i].timesignal = h_timesignal + i * npts;
        pInOutList[i].spectrum = h_spectrum + i * nstep_valid * nspec;
        pInOutList[i].nspec = nspec;
        pInOutList[i].nstep = nstep_valid;
        pInOutList[i].df = df;
        pInOutList[i].dt = delta;
    }

    // Initialize the memory for GPU
    float *d_timesignal = NULL;
    cuComplex *d_spectrum = NULL;

    float *d_segment_timesignal = NULL;
    cuComplex *d_segment_spectrum = NULL;

    float *d_filtered_segment_timesignal = NULL;
    cuComplex *d_filtered_segment_spectrum = NULL;

    cuComplex *d_filter_responses = NULL; // filter response for each segment spectrum data

    float *d_weight = NULL; // weight for each segment spectrum data
    float *d_tmp = NULL;    // used in npsmooth
    double *d_sum = NULL;   // used in rtr and rdc
    double *d_isum = NULL;  // used in rtr and rdc

    cufftHandle planfwd; // forward fft
    cufftHandle planinv; // inverse fft

    // Allocating memory in GPU Device
    allocateCudaMemoryC1(d_batch, npts, nstep_valid, nfft_2x,
                         do_runabs, wh_flag,
                         &d_timesignal,
                         &d_spectrum,
                         &d_segment_timesignal,
                         &d_segment_spectrum,
                         &d_filtered_segment_timesignal,
                         &d_filtered_segment_spectrum,
                         &d_filter_responses,
                         filter_count,
                         &d_weight, &d_tmp,
                         &d_sum, &d_isum,
                         &planfwd, &planinv);

    // ****** MAIN PROCESSING PART ****** //

    // copy each filter response to GPU and store freq_low array
    float freq_lows[filter_count];
    for (int i = 0; i < filter_count; i++)
    {
        freq_lows[i] = myResp[i].freq_low;
        CUDACHECK(cudaMemcpy2D(d_filter_responses, nfft_2x * sizeof(cuComplex), myResp[i].response, nfft_2x * sizeof(cuComplex),
                               nfft_2x * sizeof(cuComplex), 1, cudaMemcpyHostToDevice));
    }

    for (size_t h_finishcnt = 0; h_finishcnt < nValid; h_finishcnt += h_batch)
    {
        size_t h_proccnt = (h_finishcnt + h_batch > nValid) ? nValid - h_finishcnt : h_batch;

        // Setup Input Output Chain
        for (size_t i = h_finishcnt, j = 0; i < h_finishcnt + h_proccnt; i++, j++)
        {
            pInOutList[j].sacpath = InPaths.paths[i];
            pInOutList[j].specpath = OutPaths.paths[i];
        }

        ThreadPoolRead *read_pool = create_threadpool_read(MAX_THREADS);
        ThreadPoolWrite *write_pool = create_threadpool_write(MAX_THREADS);

        // Set [h_timesignal] and [h_spectrum] to zero, for input and output data
        memset(h_timesignal, 0, h_proccnt * npts * sizeof(float));
        memset(h_spectrum, 0, h_proccnt * nstep_valid * nspec * sizeof(complex));

        parallel_read_sac_c1(read_pool, h_proccnt, pInOutList, MAX_THREADS);

        // Launch GPU Processing Part
        dim3 dimgrd, dimblk;
        for (size_t d_finishcnt = 0; d_finishcnt < h_proccnt; d_finishcnt += d_batch)
        {
            size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? h_proccnt - d_finishcnt : d_batch;

            CUDACHECK(cudaMemset(d_timesignal, 0, d_proccnt * npts * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum, 0, d_proccnt * nstep_valid * nspec * sizeof(cuComplex)));

            CUDACHECK(cudaMemcpy2D(d_timesignal, npts * sizeof(float),
                                   h_timesignal + d_finishcnt * npts, npts * sizeof(float),
                                   npts * sizeof(float), d_proccnt, cudaMemcpyHostToDevice));

            for (int stepidx = 0, done_step_flag = 0; stepidx < nstep; stepidx++)
            {
                // check whether the step is done
                if (skip_step == stepidx)
                    continue;

                // clean [d_segment_timesignal] and [d_segment_spectrum]
                CUDACHECK(cudaMemset(d_segment_timesignal, 0, d_proccnt * nfft_2x * sizeof(float)));
                CUDACHECK(cudaMemset(d_segment_spectrum, 0, d_proccnt * nfft_2x * sizeof(cuComplex)));

                // copy data from [d_timesignal] to [d_segment_timesignal]
                // a segment of data in [d_segment_timesignal] is [nfft_2x] points
                CUDACHECK(cudaMemcpy2D(d_segment_timesignal, nfft_2x * sizeof(float),
                                       d_timesignal + stepidx * nseg, npts * sizeof(float),
                                       nseg * sizeof(float), d_proccnt, cudaMemcpyDeviceToDevice));

                // pre-processing check Nan, demean, detrend, TAPER_RATIO is an int < 100
                // nseg is the width of data and nfft_2x is the pitch of data
                preprocess(d_segment_timesignal, d_sum, d_isum, nseg, nfft_2x, d_proccnt, TAPER_RATIO);

                DimCompute(&dimgrd, &dimblk, nfft_2x, d_proccnt);
                if (wh_before || do_runabs)
                {
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal, (cufftComplex *)d_segment_spectrum));
                    FwdNormalize2DKernel<<<dimgrd, dimblk>>>(d_segment_spectrum, nfft_2x, nfft_2x, d_proccnt, delta);
                    if (wh_before)
                    {
                        freqWhiten(d_segment_spectrum, d_weight, d_tmp, nfft_2x, nspec, d_proccnt, nsmooth, df, freq_low_limit, freq_high_limit, FILTERFLAG);
                    }
                    if (do_onebit) // do inverse fft, since the input of onebit is time domain data
                    {
                        CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_segment_spectrum, (cufftReal *)d_segment_timesignal));
                        InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_segment_timesignal, nfft_2x, nfft_2x, d_proccnt, delta);
                    }
                }

                if (do_runabs)
                {
                    runabs_c1(d_segment_timesignal, d_segment_spectrum,
                              d_filtered_segment_timesignal, d_filtered_segment_spectrum,
                              d_weight, d_tmp, &planinv,
                              d_filter_responses, freq_lows,
                              filter_count, delta, d_proccnt, nfft_2x,
                              df, MAXVAL);
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal, (cufftComplex *)d_segment_spectrum));
                    FwdNormalize2DKernel<<<dimgrd, dimblk>>>(d_segment_spectrum, nfft_2x, nfft_2x, d_proccnt, delta);
                }

                if (do_onebit)
                {
                    onebit(d_segment_timesignal, nfft_2x, d_proccnt);
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal, (cufftComplex *)d_segment_spectrum));
                    FwdNormalize2DKernel<<<dimgrd, dimblk>>>(d_segment_spectrum, nfft_2x, nfft_2x, d_proccnt, delta);
                }

                if (!do_runabs && !do_onebit && !wh_before)
                {
                    CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_segment_timesignal, (cufftComplex *)d_segment_spectrum));
                    FwdNormalize2DKernel<<<dimgrd, dimblk>>>(d_segment_spectrum, nfft_2x, nfft_2x, d_proccnt, delta);
                }

                if (wh_after)
                {
                    freqWhiten(d_segment_spectrum, d_weight, d_tmp, nfft_2x, nspec, d_proccnt, nsmooth, df, freq_low_limit, freq_high_limit, FILTERFLAG);
                }

                // copy data from [d_segspec] to [d_specdata] DeviceToDevice
                // only the first [nfft/2+1] points are copied
                CUDACHECK(cudaMemcpy2D(d_spectrum + done_step_flag * nspec, nstep_valid * nspec * sizeof(cuComplex),
                                       d_segment_spectrum, nfft_2x * sizeof(cuComplex),
                                       nspec * sizeof(cuComplex), d_proccnt, cudaMemcpyDeviceToDevice));
                done_step_flag++;
            } // End loop of each segment
            CUDACHECK(cudaMemcpy2D(h_spectrum + d_finishcnt * nstep_valid * nspec,
                                   nstep_valid * nspec * sizeof(complex), d_spectrum,
                                   nstep_valid * nspec * sizeof(complex),
                                   nstep_valid * nspec * sizeof(complex), d_proccnt,
                                   cudaMemcpyDeviceToHost));
        } // Quit GPU processing

        // Writing the output spectra
        parallel_write_spec_c1(write_pool, h_proccnt, pInOutList, MAX_THREADS);
        destroy_threadpool_read(read_pool);
        destroy_threadpool_write(write_pool);
        // Finish processing h_proccnt
    }

    // free memory
    freeMemory(planfwd, planinv,
               &d_timesignal,
               &d_spectrum,
               &d_segment_timesignal,
               &d_segment_spectrum,
               &d_filtered_segment_timesignal,
               &d_filtered_segment_spectrum,
               &d_filter_responses,
               &d_weight, &d_tmp,
               &d_sum, &d_isum,
               &h_timesignal, &h_spectrum,
               pInOutList);
}
