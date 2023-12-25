#include "cuda.xc_dual.cuh"
#include "cuda.util.cuh"
#include "segspec.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <linux/limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

extern "C"
{
#include "sac.h"
#include "arguproc.h"
#include "read_segspec.h"
#include "read_spec_lst.h"
#include "gen_pair_dual.h"
#include "gen_ccfpath.h"
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

//  pthread_mutex_t g_paramlock;
void *writethrd(void *arg)
{
  struct timespec req, rem;
  req.tv_sec = 0;
  req.tv_nsec = 1000;
  rem.tv_sec = 0;
  rem.tv_nsec = 0;

  size_t writecnt = 0;
  SHAREDITEM *pItem = (SHAREDITEM *)arg;

  size_t batchload = 0;
  size_t totalload = 0;

  pthread_mutex_lock(&g_paramlock);
  batchload = g_batchload;
  totalload = g_totalload;
  pthread_mutex_unlock(&g_paramlock);

  while (writecnt < batchload)
  {
    for (size_t i = 0; i < totalload; i++)
    {
      SHAREDITEM *ptr = pItem + i;
      pthread_mutex_lock(&(ptr->mtx));
      if (ptr->valid == 0)
      {
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
  
  ARGUTYPE argument;
  ArgumentProcess(argc, argv, &argument);
  ARGUTYPE *parg = &argument;

  SPECNODE *pSpecSrcList, *pSpecStaList;
  PAIRNODE *pPairList;

  /* Argumnet parameter */
  float cclength = parg->cclength;
  char *ncf_dir = parg->ncf_dir;
  int gpu_id = parg->gpu_id;
  CUDACHECK(cudaSetDevice(gpu_id));

  // Generate list of input src/sta spectrum
  FilePaths *pSrcPaths = read_spec_lst(parg->src_spectrum_lst);
  FilePaths *pStaPaths = read_spec_lst(parg->sta_spectrum_lst);

  size_t srccnt = pSrcPaths->count;
  size_t stacnt = pStaPaths->count;

  SEGSPEC spechead;
  read_spechead(pSrcPaths->paths[0], &spechead);
  int nspec = spechead.nspec;
  int nstep = spechead.nstep;
  float delta = spechead.dt;
  int nfft = 2 * (nspec - 1);

  /* get npts of ouput NCF from -cclength to cclength */
  int nhalfcc = (int)floorf(cclength / delta);
  int ncc = 2 * nhalfcc + 1;
  /*********    END OF PRE DEFINING  AND PARSING    ***********/

  /* Alloc static CPU memory */
  complex *src_buffer = NULL; // input src spectrum
  complex *sta_buffer = NULL; // input sta spectrum
  float *ncf_buffer = NULL;   // output ncf data

  size_t total_cnt = 0;
  total_cnt = srccnt + stacnt;

  size_t vec_cnt = nstep * nspec;              // number of point in a spectrum file
  size_t vec_size = vec_cnt * sizeof(complex); // size of a spectrum file
  // least size of CPU memory required
  size_t fixedCpuRam = total_cnt * vec_size                  // spectrum data buffer
                       + total_cnt * sizeof(SPECNODE)        // spectrum node
                       + srccnt * stacnt * sizeof(PAIRNODE); // pair node

  /* The unitCpuram represent the memory used to write out file */
  size_t unitCpuram = nfft * sizeof(float) + sizeof(SHAREDITEM);
  size_t h_batch = EstimateCpuBatch(fixedCpuRam, unitCpuram);

  // allocate CPU memory for spectrum node and pair node
  CpuMalloc((void **)&pSpecSrcList,
            srccnt * sizeof(SPECNODE)); // src spectrum node
  CpuMalloc((void **)&pSpecStaList,
            stacnt * sizeof(SPECNODE)); // sta spectrum node
  CpuMalloc((void **)&pPairList,
            srccnt * stacnt * sizeof(PAIRNODE)); // pair node
  
  // Allocate CPU memory for spectrum data buffer
  CpuMalloc((void **)&src_buffer, srccnt * vec_size); // src spectrum data buffer
  CpuMalloc((void **)&sta_buffer, stacnt * vec_size);

  // init src spectrum node, mapping .pdata point to data Buffer
  for (size_t i = 0; i < srccnt; i++)
  {
    pSpecSrcList[i].pdata = src_buffer + i * nstep * nspec;
  }
 
  for (size_t i = 0; i < stacnt; i++)
  {
    pSpecStaList[i].pdata = sta_buffer + i * nstep * nspec;
  }
  // reading data from filenode_list to speclist.pdata
  // spec.pdata has already been mapped to srcBuffer/staBuffer

  GenSpecArray(pSrcPaths, pSpecSrcList);
  GenSpecArray(pStaPaths, pSpecStaList);

  size_t paircnt = GeneratePair_dual(pPairList, pSpecSrcList, srccnt, pSpecStaList, stacnt);
  h_batch = (h_batch > paircnt) ? paircnt : h_batch;
  
  /* Alloc cpu dynamic memory */
  CpuMalloc((void **)&ncf_buffer, h_batch * nfft * sizeof(float));

  // Set the head of output NCF of each pair src file and sta file
  for (size_t i = 0; i < paircnt; i++)
  {
    SACHEAD *phd_ncf = &(pPairList[i].headncf);
    SEGSPEC *phd_src = &(pSpecSrcList[pPairList[i].srcidx].head);
    SEGSPEC *phd_sta = &(pSpecStaList[pPairList[i].staidx].head);
    SacheadProcess(phd_ncf, phd_src, phd_sta, delta, ncc, cclength);
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

  size_t unitgpuram = sizeof(PAIRNODE)               // input pair node
                      + 2 * nfft * sizeof(complex)   // input src spectrum
                      + 2 * nfft * sizeof(float);    // output ncf data
  size_t fixedGpuRam = total_cnt * vec_size;

  // Estimate the maximum number of batch
  size_t d_batch = EstimateGpuBatch(gpu_id, fixedGpuRam, unitgpuram, numType,
                                    rank, n, inembed, istride, idist, onembed,
                                    ostride, odist, typeArr);
  // set the maximum number of batch
  d_batch = (d_batch > h_batch) ? h_batch : d_batch;

  // Define GPU memory pointer
  cuComplex *d_src_spectrum = NULL;         // input src spectrum
  cuComplex *d_sta_spectrum = NULL;         // input sta spectrum
  cuComplex *d_segment_ncf_spectrum = NULL; // output ncf data, segment in spectrum
  cuComplex *d_total_ncf_spectrum = NULL;   // output ncf data, total in spectrum
  float *d_total_ncf = NULL;                // output ncf data, time signal
  PAIRNODE *d_pairlist = NULL;              // pair node

  // Allocate GPU memory for spectrum node data buffer for input
  GpuMalloc((void **)&d_src_spectrum, srccnt * vec_size);
  GpuMalloc((void **)&d_sta_spectrum, stacnt * vec_size);

  // Copy spectrum data from CPU buffer to GPU
  CUDACHECK(cudaMemcpy(d_src_spectrum, src_buffer, srccnt * vec_size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_sta_spectrum, sta_buffer, stacnt * vec_size, cudaMemcpyHostToDevice));

  /* Alloc gpu dynamic memory with d_batch */
  CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);

  GpuMalloc((void **)&d_pairlist, d_batch * sizeof(PAIRNODE));
  GpuMalloc((void **)&d_segment_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf, d_batch * nfft * sizeof(float));

  size_t globalidx = 0;
  printf("[INFO]: Doing Cross Correlation!\n");
  for (size_t h_finishcnt = 0; h_finishcnt < paircnt; h_finishcnt += h_batch)
  {
    // Set the number of [h_proccnt]: how many ncfs will be written to disk
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

      CUDACHECK(cudaMemcpy(d_pairlist, pPairList + h_finishcnt + d_finishcnt,
                           d_proccnt * sizeof(PAIRNODE),
                           cudaMemcpyHostToDevice));

      CUDACHECK(cudaMemset(d_total_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex)));
      dim3 dimgrd, dimblk;
      DimCompute(&dimgrd, &dimblk, nspec, d_proccnt);
      for (size_t stepidx = 0; stepidx < nstep; stepidx++)
      {
        /* step by step cc */
        /* Reset temp ncf to zero */
        CUDACHECK(cudaMemset(d_segment_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex)));

        cmuldual2DKernel<<<dimgrd, dimblk>>>(d_src_spectrum, vec_cnt, stepidx * nspec,
                                             d_sta_spectrum, vec_cnt, stepidx * nspec,
                                             d_pairlist, d_proccnt, d_segment_ncf_spectrum, nfft, nspec);
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
                     pSpecSrcList[(pPairList + globalidx)->srcidx].filepath,
                     pSpecStaList[(pPairList + globalidx)->staidx].filepath,
                     ncf_dir);
          ptr->phead = &((pPairList + globalidx)->headncf);
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

  CpuFree((void **)&pItem);

  CpuFree((void **)&src_buffer);
  CpuFree((void **)&sta_buffer);
  CpuFree((void **)&ncf_buffer);

  CpuFree((void **)&pSpecSrcList);
  CpuFree((void **)&pSpecStaList);
  CpuFree((void **)&pPairList);

  // Free gpu memory
  GpuFree((void **)&d_src_spectrum);
  GpuFree((void **)&d_sta_spectrum);
  GpuFree((void **)&d_segment_ncf_spectrum);
  GpuFree((void **)&d_total_ncf_spectrum);
  GpuFree((void **)&d_total_ncf);

  CUFFTCHECK(cufftDestroy(plan));
  freeFilePaths(pSrcPaths);
  freeFilePaths(pStaPaths);

  return 0;
}
