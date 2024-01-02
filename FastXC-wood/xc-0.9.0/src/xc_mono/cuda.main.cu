#include "cuda.xc_mono.cuh"
#include "cuda.util.cuh"
#include "segspec.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
#include "sac.h"
#include "gen_ccfpath.h"
#include "complex.h"
#include "arguproc.h"
#include "read_spec_lst.h"
#include "read_segspec.h"
#include "gen_pair_mono.h"
#include "util.h"
}

pthread_mutex_t g_paramlock = PTHREAD_MUTEX_INITIALIZER;
size_t g_batchload = 0;
size_t g_totalload = 0;

typedef struct
{
  pthread_mutex_t mtx;
  int valid; /* -1: default; 1: ready to file; 2: finish to file */
  char fname[PATH_MAX];
  SACHEAD *phead;
  float *pdata;
} SHAREDITEM;

void *writethrd(void *arg)
{
  // Initializes timespec
  struct timespec req, rem;
  req.tv_sec = 0;
  req.tv_nsec = 1000;
  rem.tv_sec = 0;
  rem.tv_nsec = 0;

  // Get the parameters
  size_t writecnt = 0;
  SHAREDITEM *pItem = (SHAREDITEM *)arg;

  size_t batchload = 0;
  size_t totalload = 0;

  pthread_mutex_lock(&g_paramlock);
  batchload = g_batchload;
  totalload = g_totalload;
  pthread_mutex_unlock(&g_paramlock);

  // Loop write file, control the number of times the thread executes
  while (writecnt < batchload)
  {
    for (size_t i = 0; i < totalload; i++)
    {
      SHAREDITEM *ptr = pItem + i;
      pthread_mutex_lock(&(ptr->mtx));
      // The status of the shared item is not written
      if (ptr->valid == 0)
      {
        // `write_sac` function is used to write the SACHEAD and data of the SAC file to the file
        if (write_sac(ptr->fname, *(ptr->phead), ptr->pdata) != 0)
        {
          fprintf(stderr, "ERROR Write output NCF %s error\n", ptr->fname);

          pthread_mutex_unlock(&(ptr->mtx));

          exit(-1);
        }

        ptr->valid = 1;
        writecnt++;
      }
      pthread_mutex_unlock(&(ptr->mtx));
    }
    nanosleep(&req, &rem);
  }
  return NULL;
}

int main(int argc, char **argv)
{
  // Parse cmd arguments
  ARGUTYPE argument;
  ArgumentProcess(argc, argv, &argument);
  ARGUTYPE *parg = &argument;

  float cclength = parg->cclength;
  char *ncf_dir = parg->ncf_dir;
  int xcorr = parg->xcorr;
  int gpu_id = parg->gpu_id;

  // Set the current device to the selected GPU
  CUDACHECK(cudaSetDevice(gpu_id));

  // Generate list of input src/sta spectrum
  FilePaths *SpecPaths = read_spec_lst(parg->spectrum_lst);

  SPECNODE *pspeclist;
  PAIRNODE *ppairlist;

  size_t spec_cnt = SpecPaths->count;

  SEGSPEC spechead;
  read_spechead(SpecPaths->paths[0], &spechead);
  // parameters required for cross correlation
  int nspec = spechead.nspec;
  int nstep = spechead.nstep;
  float delta = spechead.dt;
  int nfft = 2 * (nspec - 1);

  /* get npts of ouput NCF from -cclength to cclength */
  int nhalfcc = floorf((cclength / delta) + 1e-7);
  int ncc = 2 * nhalfcc + 1;
  printf("[INFO]: cclength = %f\n", cclength);
  printf("[INFO]: delta = %f\n", delta);
  printf("[INFO]: ncc = %d\n", ncc);
  /*********    END OF PRE DEFINING  AND PARSING    ***********/

  // Alloc static CPU memory
  complex *spectrum_buffer = NULL;
  float *ncf_buffer = NULL;

  size_t vec_cnt = nstep * nspec;              // number of point in a spectrum file
  size_t vec_size = vec_cnt * sizeof(complex); // bytes size of data in spectrum file

  // least size of CPU memory required
  size_t fixedCpuRam = spec_cnt * vec_size            // spectrum data buffer
                       + spec_cnt * sizeof(SPECNODE); // spectrum, redundant

  // The unitCpuram represent the memory used to write out file
  size_t unitCpuram = nfft * sizeof(float) + sizeof(SHAREDITEM);
  size_t h_batch = EstimateCpuBatch(fixedCpuRam, unitCpuram);

  // allocate CPU memory for spectrum node and pair node
  CpuMalloc((void **)&pspeclist, spec_cnt * sizeof(SPECNODE));
  CpuMalloc((void **)&ppairlist, spec_cnt * spec_cnt * sizeof(PAIRNODE));

  // Allocate CPU memory for spectrum data buffer
  CpuMalloc((void **)&spectrum_buffer, spec_cnt * vec_size);

  // init src spectrum node, mapping .pdata point to data Buffer
  for (int i = 0; i < spec_cnt; i++)
  {
    pspeclist[i].pdata = spectrum_buffer + i * nstep * nspec;
  }

  // reading data from [SpecPaths] to [pspeclist.pdata]
  GenSpecArray(SpecPaths, pspeclist); // Find something error
  size_t paircnt = GeneratePair(ppairlist, pspeclist, spec_cnt, xcorr);
  h_batch = (h_batch > paircnt) ? paircnt : h_batch;
  // Alloc CPU output memory
  CpuMalloc((void **)&ncf_buffer, h_batch * nfft * sizeof(float));

  // Set the head of output NCF
  for (size_t i = 0; i < paircnt; i++)
  {
    SACHEAD *phdncf = &(ppairlist[i].headncf);
    SEGSPEC *phd_src = &(pspeclist[ppairlist[i].srcidx].head);
    SEGSPEC *phd_sta = &(pspeclist[ppairlist[i].staidx].head);
    SacheadProcess(phdncf, phd_src, phd_sta, delta, ncc, cclength);
  }

  /* Slave thread  property */

  SHAREDITEM *pItem;
  CpuMalloc((void **)&pItem, paircnt * sizeof(SHAREDITEM));
  for (size_t i = 0; i < paircnt; i++)
  {
    SHAREDITEM *ptr = pItem + i;
    pthread_mutex_init(&ptr->mtx, NULL);
    pthread_mutex_lock(&ptr->mtx);
    ptr->valid = -1;
    pthread_mutex_unlock(&ptr->mtx);
  }

  /* Alloc gpu static memory */
  // cufft handle
  cufftHandle plan;
  int rank = 1;
  int n[1] = {nfft};
  int inembed[1] = {nfft};
  int onembed[1] = {nfft};
  int istride = 1;
  int idist = nfft;
  int ostride = 1;
  int odist = nfft;
  cufftType type = CUFFT_C2R;
  int numType = 1;
  cufftType typeArr[1] = {type};

  size_t unitgpuram = sizeof(PAIRNODE)            // input pair node
                      + nfft * sizeof(complex)    // input src spectrum
                      + 2 * nfft * sizeof(float); // output ncf data
  size_t fixedGpuRam = spec_cnt * vec_size;

  // Estimate the maximum number of batch
  size_t d_batch = EstimateGpuBatch(gpu_id, fixedGpuRam, unitgpuram, numType,
                                    rank, n, inembed, istride, idist, onembed,
                                    ostride, odist, typeArr);
  // set the maximum number of batch
  d_batch = (d_batch > h_batch) ? h_batch : d_batch;

  // Define GPU memory pointer
  cuComplex *d_spectrum = NULL;             // input spectrum
  cuComplex *d_segment_ncf_spectrum = NULL; // output ncf data
  cuComplex *d_total_ncf_spectrum = NULL;   // output ncf data
  float *d_total_ncf = NULL;                // output ncf data
  PAIRNODE *d_pairlist = NULL;              // pair node

  // Allocate GPU memory for spectrum node data buffer for input
  GpuMalloc((void **)&d_spectrum, spec_cnt * vec_size);

  // Copy spectrum data from CPU buffer to GPU
  CUDACHECK(cudaMemcpy(d_spectrum, spectrum_buffer, spec_cnt * vec_size, cudaMemcpyHostToDevice));

  // Alloc gpu dynamic memory with d_batch
  CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);
  GpuMalloc((void **)&d_pairlist, d_batch * sizeof(PAIRNODE));
  GpuMalloc((void **)&d_segment_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf, d_batch * nfft * sizeof(float));

  size_t globalidx = 0;
  printf("[INFO]: Doing Cross Correlation!\n");
  for (size_t h_finishcnt = 0; h_finishcnt < paircnt; h_finishcnt += h_batch)
  {
    size_t h_proccnt = (h_finishcnt + h_batch > paircnt) ? (paircnt - h_finishcnt) : h_batch;

    // Set the memory of [ncfBuffer] to zero
    memset(ncf_buffer, 0, h_batch * nfft * sizeof(float));

    pthread_mutex_lock(&g_paramlock);   // lock
    g_totalload = paircnt;              // total number of pairs
    g_batchload = h_proccnt;            // number of pairs in this batch
    pthread_mutex_unlock(&g_paramlock); // unlock

    pthread_t tid;
    pthread_create(&tid, NULL, writethrd, (void *)pItem);

    // Launch GPU processing
    for (size_t d_finishcnt = 0; d_finishcnt < h_proccnt; d_finishcnt += d_batch)
    {
      cudaMemset(d_total_ncf, 0, d_batch * nfft * sizeof(float));

      size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? (h_proccnt - d_finishcnt) : d_batch;

      CUDACHECK(cudaMemcpy(d_pairlist, ppairlist + h_finishcnt + d_finishcnt,
                           d_proccnt * sizeof(PAIRNODE),
                           cudaMemcpyHostToDevice));

      CUDACHECK(cudaMemset(d_total_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex)));
      dim3 dimgrd, dimblk;
      DimCompute(&dimgrd, &dimblk, nspec, d_proccnt);
      for (size_t stepidx = 0; stepidx < nstep; stepidx++)
      {
        /* step by step cc */
        /* Reset segment ncf spectrum to zero */
        cudaMemset(d_segment_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex));

        cmulmono2DKernel<<<dimgrd, dimblk>>>(d_spectrum,
                                             vec_cnt, stepidx * nspec,
                                             vec_cnt, stepidx * nspec,
                                             d_pairlist, d_proccnt, d_segment_ncf_spectrum,
                                             nfft, nspec);
        csum2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_spectrum, nfft, d_segment_ncf_spectrum, nfft, nspec, d_proccnt, nstep);
      }
      cufftExecC2R(plan, (cufftComplex *)d_total_ncf_spectrum, (cufftReal *)d_total_ncf);
      DimCompute(&dimgrd, &dimblk, nfft, d_proccnt);
      InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_total_ncf, nfft, nfft, d_proccnt, delta);

      cudaMemcpy(ncf_buffer + d_finishcnt * nfft, d_total_ncf, d_proccnt * nfft * sizeof(float), cudaMemcpyDeviceToHost);

      for (size_t i = 0; i < d_proccnt; i++)
      {
        SHAREDITEM *ptr = pItem + globalidx;
        pthread_mutex_lock(&(ptr->mtx));
        if (ptr->valid == -1)
        {
          GenCCFPath(ptr->fname,
                     pspeclist[(ppairlist + globalidx)->srcidx].filepath,
                     pspeclist[(ppairlist + globalidx)->staidx].filepath,
                     ncf_dir);
          ptr->phead = &((ppairlist + globalidx)->headncf);
          ptr->pdata = ncf_buffer + (d_finishcnt + i) * nfft + nspec - nhalfcc - 1;
          ptr->valid = 0;
        }
        pthread_mutex_unlock(&(ptr->mtx));
        globalidx++;
      }
    }

    pthread_join(tid, NULL);
  }

  /* Free cpu memory */

  for (size_t i = 0; i < paircnt; i++)
  {
    pthread_mutex_destroy(&((pItem + i)->mtx));
  }

  printf("[INFO]: Finish Cross Correlation!\n");

  CpuFree((void **)&ncf_buffer);
  CpuFree((void **)&spectrum_buffer);
  CpuFree((void **)&pItem);
  CpuFree((void **)&pspeclist);
  CpuFree((void **)&ppairlist);

  // Free gpu memory
  GpuFree((void **)&d_spectrum);
  GpuFree((void **)&d_segment_ncf_spectrum);
  GpuFree((void **)&d_total_ncf_spectrum);
  GpuFree((void **)&d_total_ncf);
  GpuFree((void **)&d_pairlist);
  CUFFTCHECK(cufftDestroy(plan));
  freeFilePaths(SpecPaths);

  return 0;
}
