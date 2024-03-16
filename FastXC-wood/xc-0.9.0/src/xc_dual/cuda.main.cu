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
#include <errno.h>

#define K_LEN_8 8
#define K_LEN_16 16

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

// NOTE: sharedItem
typedef struct
{
  pthread_mutex_t mtx;
  int valid; /* -1: default; 1: ready to file; 2: finish to file */
  char fname[PATH_MAX];
  SACHEAD *phead;
  float *pdata;
} SHAREDITEM;

int create_parent_dir(const char *path)
{
    char *path_copy = strdup(path);
    char *parent_dir = dirname(path_copy);

    if (access(parent_dir, F_OK) == -1)
    {
        create_parent_dir(parent_dir);
        if (mkdir(parent_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST)
        {
            free(path_copy);
            return -1;
        }
    }

    free(path_copy);
    return 0;
}

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

  // tag: for debug, check batchload, totalload
  printf("[INFO]: batchload: %ld\n", batchload);
  printf("[INFO]: totalload: %ld\n", totalload);

  while (writecnt < batchload)
  {
    // FIXME: now only write 1 pair
    for (size_t i = 0; i < totalload; i++)
    {
      SHAREDITEM *ptr = pItem + i;
      pthread_mutex_lock(&(ptr->mtx));
      if (ptr->valid == 0)
      {
        // tag: for debug, check ptr->fname
        // printf("[INFO]: ptr->fname: %s\n", ptr->fname);
        // printf("[INFO]: ptr->phead->npts: %d\n", ptr->phead->npts);

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

void *writethrd_onlyOnePair(void *arg)
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

  // tag: for debug, check batchload, totalload
  printf("[INFO]: batchload: %ld\n", batchload);
  printf("[INFO]: totalload: %ld\n", totalload);

  
  for (size_t i = 0; i < 2; i++)
  {
    SHAREDITEM *ptr = pItem + i;
    pthread_mutex_lock(&(ptr->mtx));
    if (ptr->valid == 0)
    {
      // tag: for debug, check ptr->fname
      printf("[INFO]: ptr->fname: %s\n", ptr->fname);
      printf("[INFO]: ptr->phead->npts: %d\n", ptr->phead->npts);

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
  // TODO: add parameter for output stack path
  char *stack_dir = parg->stack_dir;
  CUDACHECK(cudaSetDevice(gpu_id));

  // Generate list of input src/sta spectrum
  // DONE: rewrite `read_spec_list` to read a station's all year spectrum
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
  total_cnt = srccnt + stacnt;  // NOTE: total_cnt means total number of spectrum files

  size_t vec_cnt = nstep * nspec;              // number of point in a spectrum file
  size_t vec_size = vec_cnt * sizeof(complex); // size of a spectrum file

  // TODO: here xc num is n*n, but now only 2 station, so xc num is n
  /*
  // least size of CPU memory required
  size_t fixedCpuRam = total_cnt * vec_size                  // spectrum data buffer
                       + total_cnt * sizeof(SPECNODE)        // spectrum node
                       + srccnt * stacnt * sizeof(PAIRNODE); // pair node
  */
  // least size of CPU memory required
  size_t fixedCpuRam = total_cnt * vec_size                  // spectrum data buffer
                       + total_cnt * sizeof(SPECNODE)        // spectrum node
                       + std::min(srccnt, stacnt) * sizeof(PAIRNODE); // pair node

  // tag: check fixedCpuRam for debug
  printf("[INFO]: fixedCpuRam: %.3f GB\n", (float)fixedCpuRam / (1024 * 1024 * 1024));

  /* The unitCpuram represent the memory used to write out file */
  size_t unitCpuram = nfft * sizeof(float) + sizeof(SHAREDITEM);

  // tag: check unitCpuram for debug
  printf("[INFO]: unitCpuram: %.3f MB\n", (float)unitCpuram / (1024 * 1024));

  size_t h_batch = EstimateCpuBatch(fixedCpuRam, unitCpuram);

  // tag: check h_batch for debug
  printf("[INFO]: h_batch: %ld\n", h_batch);

  // allocate CPU memory for spectrum node and pair node
  CpuMalloc((void **)&pSpecSrcList,
            srccnt * sizeof(SPECNODE)); // src spectrum node
  CpuMalloc((void **)&pSpecStaList,
            stacnt * sizeof(SPECNODE)); // sta spectrum node
  // TODO: here xc num is n*n, but now only 2 station, so xc num is n
  /*
  CpuMalloc((void **)&pPairList,
            srccnt * stacnt * sizeof(PAIRNODE)); // pair node
  */
  CpuMalloc((void **)&pPairList,
            std::min(srccnt, stacnt) * sizeof(PAIRNODE)); // pair node
  
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


  // DONE: GeneratePair_dual() need to be fixed, add filenameDate cmp
  size_t paircnt = GeneratePair_dual(pPairList, pSpecSrcList, srccnt, pSpecStaList, stacnt);

  // tag: for debug, check paircnt
  printf("[INFO]: paircnt: %ld\n", paircnt);

  h_batch = (h_batch > paircnt) ? paircnt : h_batch;
  
  /* Alloc cpu dynamic memory */
  CpuMalloc((void **)&ncf_buffer, h_batch * nfft * sizeof(float));
  memset(ncf_buffer, 0, h_batch * nfft * sizeof(float));

  // Set the head of output NCF of each pair src file and sta file
  for (size_t i = 0; i < paircnt; i++)
  {
    SACHEAD *phd_ncf = &(pPairList[i].headncf);
    SEGSPEC *phd_src = &(pSpecSrcList[pPairList[i].srcidx].head);
    SEGSPEC *phd_sta = &(pSpecStaList[pPairList[i].staidx].head);
    SacheadProcess(phd_ncf, phd_src, phd_sta, delta, ncc, cclength);
  }

  // tag: for debug, check for pPairList.size
  printf("[INFO]: pPairList.size: %ld\n", sizeof(pPairList));

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

  // TODO: now we need stack process after xc, so we need more CPU memory
  // ---------------------------stack memory-------------------------------------------
  SACHEAD template_hd = sac_null;

  size_t nstack = 0;
  size_t k = 0;

  size_t ncf_num = paircnt;
  template_hd = pPairList[0].headncf;

  char *ncf_filepath = GetNcfPath(pSpecSrcList[(pPairList + 0)->srcidx].filepath,
                     pSpecStaList[(pPairList + 0)->staidx].filepath,
                     ncf_dir);

  char template_path[256];
  strcpy(template_path, ncf_filepath);
  char *base_name = basename(template_path); 
  char *base_name_copy = strdup(base_name);

  /* Extract the required fields */
  char *fields[5];
  int i = 0;
  char *token = strtok(base_name, ".");
  while (token != NULL)
  {
    fields[i++] = token;
    token = strtok(NULL, ".");
  }

  // NOTE: filename's 1st part is sta-pair, 2nd part is component-pair
  // example: AAKH-ABNH.U-U.sac
  char *sta_pair = fields[0];
  char *component_pair = fields[1];

  char *sta_pair_copy = strdup(sta_pair);

  char *rest = sta_pair;
  char *saveptr;

  token = strtok_r(rest, "-", &saveptr);
  char *kevnm = strtok(sta_pair, "-");
  rest = NULL;
  char *kstnm = strtok_r(rest, "-", &saveptr);

  // tag: for debug, check for npts, ncf_num, ncf_filepath, kevnm, kstnm = AAKH, ABNH, component_pair = U-U
  printf("[INFO]: template_hd.npts: %d\n", template_hd.npts);
  printf("[INFO]: ncf_num: %ld\n", ncf_num);
  printf("[INFO]: ncf_filepath: %s\n", ncf_filepath);
  // printf("[INFO]: kevnm: %s, kstnm: %s\n", kevnm, kstnm);
  // printf("[INFO]: component_pair: %s\n", component_pair);

  /* Write fields to the sac header */
  strncpy(template_hd.kstnm, kstnm, K_LEN_8);
  strncpy(template_hd.kevnm, kevnm, K_LEN_16);
  strncpy(template_hd.kcmpnm, component_pair, K_LEN_8);

  int npts = template_hd.npts;
  SACHEAD hdstack = template_hd;

  /* change the reference time nzyear nzjday nzhour nzmin nzsec nzmsec */
  hdstack.nzyear = 2010;
  hdstack.nzjday = 214;
  hdstack.nzhour = 16;
  hdstack.nzmin = 0;
  hdstack.nzsec = 0;
  hdstack.nzmsec = 0;

  /* Copy coordinate infomation from first sac file */
  hdstack.stla = template_hd.stla;
  hdstack.stlo = template_hd.stlo;
  hdstack.evla = template_hd.evla;
  hdstack.evlo = template_hd.evlo;

  hdstack.dist = template_hd.dist;
  hdstack.az = template_hd.az;
  hdstack.baz = template_hd.baz;
  hdstack.gcarc = template_hd.gcarc;

  float *stackcc = NULL;
  stackcc = (float *)malloc(sizeof(float) * npts);
  nstack = 0;

  // set stackcc to zero
  for (k = 0; k < npts; k++)
  {
    stackcc[k] = 0.0;
  }

  // NOTE: create stack dir
  char *out_sac = createFilePath(stack_dir, sta_pair_copy, base_name_copy);

  // ---------------------------stack memory-------------------------------------------
  
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

  // tag: for debug, check vec_size, unitgpuram, fixedGpuRam
  printf("[INFO]: -----------------------------GPU Alloc Start-----------------------------------------\n");
  printf("[INFO]: total_cnt: %ld\n", total_cnt);
  printf("[INFO]: vec_size: %.3f MB\n", (float)vec_size / (1024 * 1024));
  printf("[INFO]: unitgpuram: %.3f MB\n", (float)unitgpuram / (1024 * 1024));
  printf("[INFO]: fixedGpuRam: %.3f GB\n", (float)fixedGpuRam / (1024 * 1024 * 1024));

  // FIXME: if avail_gpu_ram < fixedGpuRam, we need to 
  size_t gpu_avail_ram = QueryAvailGpuRam(gpu_id);
  size_t batch_data_unit_count = gpu_avail_ram / ( 2.5 * vec_size );  // 1 for sta, 1 for src, 0.5 for others
  size_t total_batches = (srccnt + batch_data_unit_count - 1) / batch_data_unit_count;

  // tag: for debug, check batch_data_unit_count
  printf("[INFO]: batch_data_unit_count: %ld\n", batch_data_unit_count);
  printf("[INFO]: total_batches: %ld\n", total_batches);

  size_t globalidx_batch = 0;
  size_t h_finishcnt = 0;
  size_t all_finishcnt = 0;

  size_t fixedGpuRam_for_batch = batch_data_unit_count * vec_size * 2;
  // tag: for debug, check fixedGpuRam_for_batch
  printf("[INFO]: fixedGpuRam_for_batch: %.3f GB\n", (float)fixedGpuRam_for_batch / (1024 * 1024 * 1024));

  size_t d_batch = EstimateGpuBatch(gpu_id, fixedGpuRam_for_batch, unitgpuram, numType,
                                      rank, n, inembed, istride, idist, onembed,
                                      ostride, odist, typeArr);
  d_batch = (d_batch > h_batch) ? h_batch : d_batch;
  // tag: for debug, check d_batch
  printf("[INFO]: d_batch: %ld\n", d_batch);

  // tag: starttime, gpu_alloc_start_time
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  struct timespec gpu_alloc_start_time, gpu_alloc_end_time;
  clock_gettime(CLOCK_MONOTONIC, &gpu_alloc_start_time);

  // FIXME: out of for loop, GPUMalloc
  // Define GPU memory pointer for each batch
  cuComplex *d_src_spectrum_batch = NULL;         // input src spectrum
  cuComplex *d_sta_spectrum_batch = NULL;         // input sta spectrum
  cuComplex *d_segment_ncf_spectrum_batch = NULL; // output ncf data, segment in spectrum
  cuComplex *d_total_ncf_spectrum_batch = NULL;   // output ncf data, total in spectrum
  float *d_total_ncf_batch = NULL;                // output ncf data, time signal
  PAIRNODE *d_pairlist_batch = NULL;              // pair node

  GpuMalloc((void **)&d_src_spectrum_batch, batch_data_unit_count * vec_size);
  GpuMalloc((void **)&d_sta_spectrum_batch, batch_data_unit_count * vec_size);

  // tag: for debug, check batch_data_unit_count * vec_size
  size_t sta_spectrum_size = batch_data_unit_count * vec_cnt;

  GpuMalloc((void **)&d_pairlist_batch, d_batch * sizeof(PAIRNODE));
  GpuMalloc((void **)&d_segment_ncf_spectrum_batch, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_spectrum_batch, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_batch, d_batch * nfft * sizeof(float));

  // tag: for debug, check each's size
  printf("[INFO]: d_src_spectrum_batch: %.5f GB\n", (float)batch_data_unit_count * vec_size / (1024 * 1024 * 1024));
  printf("[INFO]: d_sta_spectrum_batch: %.5f GB\n", (float)batch_data_unit_count * vec_size / (1024 * 1024 * 1024));
  printf("[INFO]: d_pairlist_batch: %.5f GB\n", (float)d_batch * sizeof(PAIRNODE) / (1024 * 1024 * 1024));
  printf("[INFO]: d_segment_ncf_spectrum_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(complex) / (1024 * 1024 * 1024));
  printf("[INFO]: d_total_ncf_spectrum_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(complex) / (1024 * 1024 * 1024));
  printf("[INFO]: d_total_ncf_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(float) / (1024 * 1024 * 1024));
  printf("[INFO]: TOTAL_GPU_SIZE: %.5f GB\n", (float)(batch_data_unit_count * vec_size + batch_data_unit_count * vec_size + d_batch * sizeof(PAIRNODE) + d_batch * nfft * sizeof(complex) + d_batch * nfft * sizeof(complex) + d_batch * nfft * sizeof(float)) / (1024 * 1024 * 1024));

  // tag: gpu_alloc_end_time
  clock_gettime(CLOCK_MONOTONIC, &gpu_alloc_end_time);
  double elapsed_gpu_alloc_time = (gpu_alloc_end_time.tv_sec - gpu_alloc_start_time.tv_sec) +
                      (gpu_alloc_end_time.tv_nsec - gpu_alloc_start_time.tv_nsec) / 1e9;
  

  printf("[INFO]: -----------------------------GPU Alloc Finish-----------------------------------------\n");

  double xc_time[total_batches];

  for(size_t gpu_batch = 0; gpu_batch < total_batches; gpu_batch++) {
    size_t start_index = gpu_batch * batch_data_unit_count;
    size_t end_index = min(start_index + batch_data_unit_count, srccnt);
    size_t current_batch_size = end_index - start_index;

    printf("[INFO]: Processing batch %ld/%ld, current_batch_size: %ld\n", gpu_batch + 1, total_batches, current_batch_size);

    // Copy spectrum data from CPU buffer to GPU
    CUDACHECK(cudaMemcpy(d_src_spectrum_batch, src_buffer + start_index, current_batch_size * vec_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sta_spectrum_batch, sta_buffer + start_index, current_batch_size * vec_size, cudaMemcpyHostToDevice));

    /* Alloc gpu dynamic memory with d_batch */
    CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);

    h_finishcnt = 0;

    printf("[INFO]: Doing Cross Correlation!\n");
    for(; h_finishcnt < current_batch_size; h_finishcnt += current_batch_size) {
      // Set the number of [h_proccnt]: how many ncfs will be written to disk
      size_t h_proccnt = (h_finishcnt + current_batch_size > paircnt) ? (paircnt - h_finishcnt) : current_batch_size;

      // tag: for debug, check h_proccnt
      printf("[INFO]: h_proccnt: %ld\n", h_proccnt);

      // Set the memory of [ncfBuffer] to zero
      // memset(ncf_buffer, 0, h_batch * nfft * sizeof(float));

      // tag: xc_start_time
      struct timespec xc_start_time, xc_end_time;
      clock_gettime(CLOCK_MONOTONIC, &xc_start_time);

      // tag: for debug, check ncf_buffer
      printf("[INFO]: ncf_buffer: %.3f MB\n", (float)h_batch * nfft * sizeof(float) / (1024 * 1024));

      size_t d_finishcnt = 0;

      // Launch GPU processing
      for(; d_finishcnt < current_batch_size; ) {
        CUDACHECK(cudaMemset(d_total_ncf_batch, 0, d_batch * nfft * sizeof(float)));

        // size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? (h_proccnt - d_finishcnt) : d_batch;
        size_t d_proccnt = (d_finishcnt + d_batch > current_batch_size) ? (current_batch_size - d_finishcnt) : d_batch;
        
        // // tag: for debug, check d_proccnt
        // printf("[INFO]: d_proccnt: %ld,\t d_batch: %ld\n", d_proccnt, d_batch);

        CUDACHECK(cudaMemcpy(d_pairlist_batch, pPairList + h_finishcnt + d_finishcnt + all_finishcnt,
                            d_proccnt * sizeof(PAIRNODE),
                            cudaMemcpyHostToDevice));

        // tag: for debug, check d_pairlist_batch
        PAIRNODE* d_pairlist_batch_copy = (PAIRNODE*)malloc(d_proccnt * sizeof(PAIRNODE));
        CUDACHECK(cudaMemcpy(d_pairlist_batch_copy, d_pairlist_batch, d_proccnt * sizeof(PAIRNODE), cudaMemcpyDeviceToHost));
        
        CUDACHECK(cudaMemset(d_total_ncf_spectrum_batch, 0, d_proccnt * nfft * sizeof(cuComplex)));
        dim3 dimgrd, dimblk;
        DimCompute(&dimgrd, &dimblk, nspec, d_proccnt);
        // NOTE: process each step, example: divide 24h into 12 steps
        for (size_t stepidx = 0; stepidx < nstep; stepidx++) {
          /* step by step cc */
          /* Reset temp ncf to zero */
          CUDACHECK(cudaMemset(d_segment_ncf_spectrum_batch, 0, d_proccnt * nfft * sizeof(cuComplex)));

          cmuldual2DKernel<<<dimgrd, dimblk>>>(d_src_spectrum_batch, vec_cnt, stepidx * nspec,
                                             d_sta_spectrum_batch, vec_cnt, stepidx * nspec,
                                             d_pairlist_batch, d_proccnt, d_segment_ncf_spectrum_batch, nfft, nspec, sta_spectrum_size);

          csum2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_spectrum_batch, nfft, d_segment_ncf_spectrum_batch, nfft, nspec, d_proccnt, nstep);
        }
        cufftExecC2R(plan, (cufftComplex *)d_total_ncf_spectrum_batch, (cufftReal *)d_total_ncf_batch);

        cudaDeviceSynchronize();

        DimCompute(&dimgrd, &dimblk, nfft, d_proccnt);
        InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_batch, nfft, nfft, d_proccnt, delta);
        CUDACHECK(cudaMemcpy(ncf_buffer + (all_finishcnt + d_finishcnt) * nfft, d_total_ncf_batch, d_proccnt * nfft * sizeof(float), cudaMemcpyDeviceToHost));

        /*
        // tag: for debug, check d_total_ncf_spectrum_batch, d_total_ncf_batch
        complex* d_total_ncf_spectrum_batch_copy = (complex*)malloc(d_proccnt * nfft * sizeof(complex));
        float* d_total_ncf_batch_copy = (float*)malloc(d_proccnt * nfft * sizeof(float));
        CUDACHECK(cudaMemcpy(d_total_ncf_spectrum_batch_copy, d_total_ncf_spectrum_batch, d_proccnt * nfft * sizeof(complex), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpy(d_total_ncf_batch_copy, d_total_ncf_batch, d_proccnt * nfft * sizeof(float), cudaMemcpyDeviceToHost));
  
        // for(int i = 0; i < d_proccnt * nfft; i++) {
        //   // if(d_total_ncf_spectrum_batch_copy[i].x != 0 || d_total_ncf_spectrum_batch_copy[i].y != 0) {
        //   //   printf("[INFO]: d_total_ncf_spectrum_batch_copy[%d]: %.9f, %.9f\n", i, d_total_ncf_spectrum_batch_copy[i].x, d_total_ncf_spectrum_batch_copy[i].y);
        //   // }
        //   if(d_total_ncf_batch_copy[i] != 0) {
        //     printf("[INFO]: d_total_ncf_batch_copy[%d]: %.9f\n", i, d_total_ncf_batch_copy[i]);
        //   }
        // }

        // // tag: for debug, check for ncf_buffer
        // for(int i = 0; i < h_batch * nfft; i++ ) {
        //   if(ncf_buffer[i] != 0) {
        //     printf("[INFO]: ncf_buffer[%d]: %.9f\n", i, ncf_buffer[i]);
        //   }
        // }
        */

        // NOTE: here cuda_calc finished
        for(size_t i = 0; i < d_proccnt; i++) {
          SHAREDITEM *ptr = pItem + globalidx_batch;
          pthread_mutex_lock(&(ptr->mtx));
          if (ptr->valid == -1) {
            ptr->phead = &((pPairList + globalidx_batch)->headncf);
            ptr->pdata = ncf_buffer + (all_finishcnt + d_finishcnt + i) * nfft + nspec - nhalfcc - 1;
            ptr->valid = 0;
          }
          pthread_mutex_unlock(&(ptr->mtx));
          globalidx_batch++;
        }

        d_finishcnt += d_proccnt;
      }

      // tag: xc_end_time
      clock_gettime(CLOCK_MONOTONIC, &xc_end_time);
      double elapsed_xc_time = (xc_end_time.tv_sec - xc_start_time.tv_sec) +
                        (xc_end_time.tv_nsec - xc_start_time.tv_nsec) / 1e9;
      xc_time[gpu_batch] = elapsed_xc_time;
    }

    all_finishcnt += current_batch_size;
    
  }

  

  /*
  // // Estimate the maximum number of batch
  // size_t d_batch = EstimateGpuBatch(gpu_id, fixedGpuRam, unitgpuram, numType,
  //                                   rank, n, inembed, istride, idist, onembed,
  //                                   ostride, odist, typeArr);
  // // set the maximum number of batch
  // d_batch = (d_batch > h_batch) ? h_batch : d_batch;

  // // tag: for debug, check d_batch
  // printf("[INFO]: d_batch: %ld\n", d_batch);

  // // Define GPU memory pointer
  // cuComplex *d_src_spectrum = NULL;         // input src spectrum
  // cuComplex *d_sta_spectrum = NULL;         // input sta spectrum
  // cuComplex *d_segment_ncf_spectrum = NULL; // output ncf data, segment in spectrum
  // cuComplex *d_total_ncf_spectrum = NULL;   // output ncf data, total in spectrum
  // float *d_total_ncf = NULL;                // output ncf data, time signal
  // PAIRNODE *d_pairlist = NULL;              // pair node

  // // Allocate GPU memory for spectrum node data buffer for input
  // GpuMalloc((void **)&d_src_spectrum, srccnt * vec_size);
  // GpuMalloc((void **)&d_sta_spectrum, stacnt * vec_size);

  // // Copy spectrum data from CPU buffer to GPU
  // CUDACHECK(cudaMemcpy(d_src_spectrum, src_buffer, srccnt * vec_size, cudaMemcpyHostToDevice));
  // CUDACHECK(cudaMemcpy(d_sta_spectrum, sta_buffer, stacnt * vec_size, cudaMemcpyHostToDevice));

  // /* Alloc gpu dynamic memory with d_batch */
  // CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);

  // GpuMalloc((void **)&d_pairlist, d_batch * sizeof(PAIRNODE));
  // GpuMalloc((void **)&d_segment_ncf_spectrum, d_batch * nfft * sizeof(complex));
  // GpuMalloc((void **)&d_total_ncf_spectrum, d_batch * nfft * sizeof(complex));
  // GpuMalloc((void **)&d_total_ncf, d_batch * nfft * sizeof(float));

  // size_t globalidx = 0;

  // // tag: starttime
  // struct timespec start_time, end_time;
  // clock_gettime(CLOCK_MONOTONIC, &start_time);

  // printf("[INFO]: Doing Cross Correlation!\n");
  // for (size_t h_finishcnt = 0; h_finishcnt < paircnt; h_finishcnt += h_batch)
  // {
  //   // Set the number of [h_proccnt]: how many ncfs will be written to disk
  //   size_t h_proccnt = (h_finishcnt + h_batch > paircnt) ? (paircnt - h_finishcnt) : h_batch;

  //   // tag: for debug, check h_proccnt
  //   printf("[INFO]: h_proccnt: %ld\n", h_proccnt);

  //   // Set the memory of [ncfBuffer] to zero
  //   memset(ncf_buffer, 0, h_batch * nfft * sizeof(float));

  //   // tag: for debug, check ncf_buffer
  //   printf("[INFO]: ncf_buffer: %.3f MB\n", (float)h_batch * nfft * sizeof(float) / (1024 * 1024));

  //   pthread_mutex_lock(&g_paramlock);   // lock
  //   g_totalload = paircnt;              // total number of pairs
  //   g_batchload = h_proccnt;            // number of pairs in this batch
  //   pthread_mutex_unlock(&g_paramlock); // unlock

  //   // FIXME: here create a new thread for write sac file
  //   // pthread_t tid;
  //   // pthread_create(&tid, NULL, writethrd, (void *)pItem);

  //   // Launch GPU processing
  //   for (size_t d_finishcnt = 0; d_finishcnt < h_proccnt; d_finishcnt += d_batch)
  //   {
  //     cudaMemset(d_total_ncf, 0, d_batch * nfft * sizeof(float));

  //     size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? (h_proccnt - d_finishcnt) : d_batch;

  //     // tag: for debug, check for d_proccnt = 10,10,...,10,5
  //     // printf("[INFO]: d_proccnt: %ld\n", d_proccnt);

  //     CUDACHECK(cudaMemcpy(d_pairlist, pPairList + h_finishcnt + d_finishcnt,
  //                          d_proccnt * sizeof(PAIRNODE),
  //                          cudaMemcpyHostToDevice));

  //     CUDACHECK(cudaMemset(d_total_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex)));
  //     dim3 dimgrd, dimblk;
  //     DimCompute(&dimgrd, &dimblk, nspec, d_proccnt);
  //     // NOTE: process each step, example: divide 24h into 12 steps
  //     for (size_t stepidx = 0; stepidx < nstep; stepidx++)
  //     {
  //       /* step by step cc */
  //       /* Reset temp ncf to zero */
  //       CUDACHECK(cudaMemset(d_segment_ncf_spectrum, 0, d_proccnt * nfft * sizeof(cuComplex)));

  //       // TODO: `cmuldual2DKernel` need to be rewrite
  //       cmuldual2DKernel<<<dimgrd, dimblk>>>(d_src_spectrum, vec_cnt, stepidx * nspec,
  //                                            d_sta_spectrum, vec_cnt, stepidx * nspec,
  //                                            d_pairlist, d_proccnt, d_segment_ncf_spectrum, nfft, nspec);

  //       // TODO: `csum2DKernel` need to be rewrite                                     
  //       csum2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_spectrum, nfft, d_segment_ncf_spectrum, nfft, nspec, d_proccnt, nstep);
  //     }
  //     cufftExecC2R(plan, (cufftComplex *)d_total_ncf_spectrum, (cufftReal *)d_total_ncf);
  //     DimCompute(&dimgrd, &dimblk, nfft, d_proccnt);
  //     InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_total_ncf, nfft, nfft, d_proccnt, delta);
  //     cudaMemcpy(ncf_buffer + d_finishcnt * nfft, d_total_ncf, d_proccnt * nfft * sizeof(float), cudaMemcpyDeviceToHost);

  //     // FIXME: here cuda_calc finished, start to stack
      
  //     for (size_t i = 0; i < d_proccnt; i++)
  //     {
  //       SHAREDITEM *ptr = pItem + globalidx;
  //       pthread_mutex_lock(&(ptr->mtx));
  //       if (ptr->valid == -1)
  //       {
  //         // tag: now not create ncf dir
  //         // GenCCFPath(ptr->fname,
  //         //            pSpecSrcList[(pPairList + globalidx)->srcidx].filepath,
  //         //            pSpecStaList[(pPairList + globalidx)->staidx].filepath,
  //         //            ncf_dir);

  //         ptr->phead = &((pPairList + globalidx)->headncf);
  //         ptr->pdata = ncf_buffer + (d_finishcnt + i) * nfft + nspec - nhalfcc - 1;
  //         ptr->valid = 0;
  //       }
  //       pthread_mutex_unlock(&(ptr->mtx));
  //       globalidx++;
  //     }

  //   }

  //   // pthread_join(tid, NULL);
  // }
  

  // TODO: stack process
  // -----------------------------------------------------------------------------
  // tag: start stack time
  struct timespec start_stack_time, end_stack_time;
  clock_gettime(CLOCK_MONOTONIC, &start_stack_time);

  for (size_t i = 0; i < ncf_num; i++) {
    for (k = 0; k < npts; k++) {
      stackcc[k] = stackcc[k] + pItem[i].pdata[k];
    }
    nstack++;
  }

  // tag: for debug, check stackcc
  int normalize = 1;

  if (normalize == 1)
  {
    for (k = 0; k < npts; k++)
    {
      stackcc[k] /= ncf_num;
    }
  }

  // tag: end stack time
  clock_gettime(CLOCK_MONOTONIC, &end_stack_time);
  double elapsed_stack_time = (end_stack_time.tv_sec - start_stack_time.tv_sec) +
                      (end_stack_time.tv_nsec - start_stack_time.tv_nsec) / 1e9;
  
  hdstack.unused27 = nstack;
  // char *out_sac = "/home/woodwood/hpc/station_2/ncf_hinet_AAKH_ABNH/stack/AAKH-ABNH/AAKH-ABNH.U-U.sac";
  char *out_sac_copy = strdup(out_sac);
  
  // for debug, check for out_sac
  printf("[INFO]: out_sac: %s\n", out_sac);

  if (create_parent_dir(out_sac) == -1)
  {
    fprintf(stderr, "Error creating directory %s: ", dirname(out_sac_copy));
    perror(NULL);
    free(out_sac_copy);
    return 1;
  }
  write_sac(out_sac, hdstack, stackcc);
  // -----------------------------------------------------------------------------

  // tag: endtime
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  for (size_t i = 0; i < paircnt; i++)
  {
    pthread_mutex_destroy(&((pItem + i)->mtx));
  }

  printf("[INFO]: Finish Cross Correlation!\n");

  free(stackcc);

  CpuFree((void **)&pItem);

  CpuFree((void **)&src_buffer);
  CpuFree((void **)&sta_buffer);
  CpuFree((void **)&ncf_buffer);

  CpuFree((void **)&pSpecSrcList);
  CpuFree((void **)&pSpecStaList);
  CpuFree((void **)&pPairList);

  /* Free cpu memory */
  // GpuFree((void **)&d_src_spectrum);
  // GpuFree((void **)&d_sta_spectrum);
  // GpuFree((void **)&d_segment_ncf_spectrum);
  // GpuFree((void **)&d_total_ncf_spectrum);
  // GpuFree((void **)&d_total_ncf);

  // Free gpu memory for each batch
  GpuFree((void **)&d_src_spectrum_batch);
  GpuFree((void **)&d_sta_spectrum_batch);
  GpuFree((void **)&d_pairlist_batch);
  GpuFree((void **)&d_segment_ncf_spectrum_batch);
  GpuFree((void **)&d_total_ncf_spectrum_batch);
  GpuFree((void **)&d_total_ncf_batch);

  CUFFTCHECK(cufftDestroy(plan));
  freeFilePaths(pSrcPaths);
  freeFilePaths(pStaPaths);

  printf("[INFO]: Gpu Alloc time: %.6f seconds\n", elapsed_gpu_alloc_time);

  double xc_time_sum = 0;
  for(int i = 0; i < total_batches; i++) {
    printf("[INFO]: XC time for batch %d: %.6f seconds\n", i, xc_time[i]);
    xc_time_sum += xc_time[i];
  }
  printf("[INFO]: XC time: %.6f seconds\n", xc_time_sum);

  printf("[INFO]: Stack time: %.6f seconds\n", elapsed_stack_time);
  printf("[INFO]: Total time: %.6f seconds\n", elapsed_time);

  return 0;
}
