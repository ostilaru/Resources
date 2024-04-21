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

int main(int argc, char **argv)
{
  // tag: starttime
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  struct timespec start_readFile_time, end_readFile_time;
  clock_gettime(CLOCK_MONOTONIC, &start_readFile_time);

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
  size_t GB = 1 << 30;
  printf("[INFO]: fixedCpuRam: %.3f GB\n", (float)fixedCpuRam / (GB));

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

  // tag: start_read_MergeFile_time
  struct timespec start_read_MergeFile_time, end_read_MergeFile_time;
  clock_gettime(CLOCK_MONOTONIC, &start_read_MergeFile_time);

  // TODO: modify read way:
  size_t src_buffer_size = srccnt * vec_size;
  size_t sta_buffer_size = stacnt * vec_size;
  const char *src_mergeFiles = "/home/woodwood/hpc/station_2/zz-mergeFinalFile/array1_2year_merged_file.segspec";
  const char *sta_mergeFiles = "/home/woodwood/hpc/station_2/zz-mergeFinalFile/array2_2year_merged_file.segspec";
  readMergedSpecBuffer_onlyOneHead(src_mergeFiles, src_buffer, src_buffer_size);
  readMergedSpecBuffer_onlyOneHead(sta_mergeFiles, sta_buffer, sta_buffer_size);

  // tag: end_read_MergeFile_time
  clock_gettime(CLOCK_MONOTONIC, &end_read_MergeFile_time);
  double elapsed_read_MergeFile_time = getElapsedTime(start_read_MergeFile_time, end_read_MergeFile_time);

  // tag: start genSpecArray_time
  struct timespec start_genSpecArray_time, end_genSpecArray_time;
  clock_gettime(CLOCK_MONOTONIC, &start_genSpecArray_time);

  GenSpecArray(pSrcPaths, pSpecSrcList);
  GenSpecArray(pStaPaths, pSpecStaList);

  // tag: end_genSpecArray_time
  clock_gettime(CLOCK_MONOTONIC, &end_genSpecArray_time);
  double elapsed_genSpecArray_time = getElapsedTime(start_genSpecArray_time, end_genSpecArray_time);

  // tag: end_readFile_time
  clock_gettime(CLOCK_MONOTONIC, &end_readFile_time);
  double elapsed_readFile_time = getElapsedTime(start_readFile_time, end_readFile_time);

  // tag: starttime
  struct timespec start_genpair_time, end_genpair_time;
  clock_gettime(CLOCK_MONOTONIC, &start_genpair_time);

  // DONE: GeneratePair_dual() need to be fixed, add filenameDate cmp
  size_t paircnt = GeneratePair_dual(pPairList, pSpecSrcList, srccnt, pSpecStaList, stacnt);

  

  // tag: for debug, check paircnt
  printf("[INFO]: paircnt: %ld\n", paircnt);

  h_batch = (h_batch > paircnt) ? paircnt : h_batch;
  printf("[INFO]: h_batch: %ld\n", h_batch);
  
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
  cufftHandle plan, c2c_plan;
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
  printf("[INFO]: fixedGpuRam: %.3f GB\n", (float)fixedGpuRam / (GB));

  // FIXME: if avail_gpu_ram < fixedGpuRam, we need to 
  size_t gpu_avail_ram = QueryAvailGpuRam(gpu_id);
  size_t batch_data_unit_count = gpu_avail_ram / ( 2.5 * vec_size );  // 1 for sta, 1 for src, 0.5 for others
  size_t total_batches = (srccnt + batch_data_unit_count - 1) / batch_data_unit_count;

  // tag: for debug, check batch_data_unit_count
  printf("[INFO]: batch_data_unit_count: %ld\n", batch_data_unit_count);
  printf("[INFO]: total_batches: %ld\n", total_batches);

  size_t globalidx_batch = 0;
  size_t all_finishcnt = 0;

  size_t fixedGpuRam_for_batch = batch_data_unit_count * vec_size * 2;
  // tag: for debug, check fixedGpuRam_for_batch
  printf("[INFO]: fixedGpuRam_for_batch: %.3f GB\n", (float)fixedGpuRam_for_batch / (GB));

  size_t d_batch = EstimateGpuBatch(gpu_id, fixedGpuRam_for_batch, unitgpuram, numType,
                                      rank, n, inembed, istride, idist, onembed,
                                      ostride, odist, typeArr);
  d_batch = (d_batch > h_batch) ? h_batch : d_batch;

  // tag: for debug, check d_batch
  printf("[INFO]: d_batch: %ld\n", d_batch);

  // tag: end_genpair_time
  clock_gettime(CLOCK_MONOTONIC, &end_genpair_time);
  double elapsed_genpair_time = getElapsedTime(start_genpair_time, end_genpair_time);

  // tag: gpu_alloc_start_time
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

  /* Alloc gpu dynamic memory with d_batch */
  CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);
  // CufftPlanAlloc(&c2c_plan, 1, n, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch);

  GpuMalloc((void **)&d_pairlist_batch, d_batch * sizeof(PAIRNODE));
  GpuMalloc((void **)&d_segment_ncf_spectrum_batch, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_spectrum_batch, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_batch, d_batch * nfft * sizeof(float));

  cuComplex *d_ncf_buffer_complex_d = NULL;
  GpuMalloc((void **)&d_ncf_buffer_complex_d, paircnt * nfft * sizeof(cuComplex));

  // tag: for debug, check each's size
  
  printf("[INFO]: d_src_spectrum_batch: %.5f GB\n", (float)batch_data_unit_count * vec_size / (GB));
  printf("[INFO]: d_sta_spectrum_batch: %.5f GB\n", (float)batch_data_unit_count * vec_size / (GB));
  printf("[INFO]: d_pairlist_batch: %.5f GB\n", (float)d_batch * sizeof(PAIRNODE) / (GB));
  printf("[INFO]: d_segment_ncf_spectrum_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(complex) / (GB));
  printf("[INFO]: d_total_ncf_spectrum_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(complex) / (GB));
  printf("[INFO]: d_total_ncf_batch: %.5f GB\n", (float)d_batch * nfft * sizeof(float) / (GB));
  printf("[INFO]: d_ncf_buffer_complex_d: %.5f GB\n", (float)paircnt * nfft * sizeof(cuComplex) / (GB));
  float d_src_spectrum_batch_mem = (float)batch_data_unit_count * vec_size / (GB);
  float d_sta_spectrum_batch_mem = (float)batch_data_unit_count * vec_size / (GB);
  float d_pairlist_batch_mem = (float)d_batch * sizeof(PAIRNODE) / (GB);
  float d_segment_ncf_spectrum_batch_mem = (float)d_batch * nfft * sizeof(complex) / (GB);
  float d_total_ncf_spectrum_batch_mem = (float)d_batch * nfft * sizeof(complex) / (GB);
  float d_total_ncf_batch_mem = (float)d_batch * nfft * sizeof(float) / (GB);
  float d_ncf_buffer_complex_d_mem = (float)paircnt * nfft * sizeof(cuComplex) / (GB);
  float batch_gpu_size = d_src_spectrum_batch_mem + d_sta_spectrum_batch_mem + d_pairlist_batch_mem + d_segment_ncf_spectrum_batch_mem + d_total_ncf_spectrum_batch_mem + d_total_ncf_batch_mem + d_ncf_buffer_complex_d_mem;
  printf("[INFO]: TOTAL_GPU_SIZE: %.5f GB\n", batch_gpu_size);

  // tag: gpu_alloc_end_time
  clock_gettime(CLOCK_MONOTONIC, &gpu_alloc_end_time);
  double elapsed_gpu_alloc_time = getElapsedTime(gpu_alloc_start_time, gpu_alloc_end_time);
  
  int B_SIZE = 1024;
  int G_SIZE = (paircnt * nfft + B_SIZE - 1) / B_SIZE;

  printf("[INFO]: -----------------------------GPU Alloc Finish-----------------------------------------\n");

  double xc_time[total_batches];

  for(size_t gpu_batch = 0; gpu_batch < total_batches; gpu_batch++) {
    size_t start_index = gpu_batch * batch_data_unit_count;
    size_t end_index = min(start_index + batch_data_unit_count, srccnt);
    size_t current_batch_size = end_index - start_index;

    printf("[INFO]: Processing batch %ld/%ld, current_batch_size: %ld\n", gpu_batch + 1, total_batches, current_batch_size);

    // tag: xc_start_time
    struct timespec xc_start_time, xc_end_time;
    clock_gettime(CLOCK_MONOTONIC, &xc_start_time);

    // Copy spectrum data from CPU buffer to GPU
    CUDACHECK(cudaMemcpy(d_src_spectrum_batch, src_buffer + start_index * vec_cnt, current_batch_size * vec_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sta_spectrum_batch, sta_buffer + start_index * vec_cnt, current_batch_size * vec_size, cudaMemcpyHostToDevice));

    printf("[INFO]: Doing Cross Correlation!\n");

    size_t d_finishcnt = 0;

    // Launch GPU processing
    while(d_finishcnt < current_batch_size) {
      CUDACHECK(cudaMemset(d_total_ncf_batch, 0, d_batch * nfft * sizeof(float)));

      // size_t d_proccnt = (d_finishcnt + d_batch > h_proccnt) ? (h_proccnt - d_finishcnt) : d_batch;
      size_t d_proccnt = (d_finishcnt + d_batch > current_batch_size) ? (current_batch_size - d_finishcnt) : d_batch;

      CUDACHECK(cudaMemcpy(d_pairlist_batch, pPairList + d_finishcnt + all_finishcnt,
                          d_proccnt * sizeof(PAIRNODE),
                          cudaMemcpyHostToDevice));

      CUDACHECK(cudaMemset(d_total_ncf_spectrum_batch, 0, d_batch * nfft * sizeof(cuComplex)));
      dim3 dimgrd, dimblk;
      DimCompute(&dimgrd, &dimblk, nspec, d_proccnt);
      // NOTE: process each step, example: divide 24h into 12 steps
      for (size_t stepidx = 0; stepidx < nstep; stepidx++) {
        /* step by step cc */
        /* Reset temp ncf to zero */
        CUDACHECK(cudaMemset(d_segment_ncf_spectrum_batch, 0, d_batch * nfft * sizeof(cuComplex)));

        cmuldual2DKernel<<<dimgrd, dimblk>>>(d_src_spectrum_batch, vec_cnt, stepidx * nspec,
                                            d_sta_spectrum_batch, vec_cnt, stepidx * nspec,
                                            d_pairlist_batch, d_proccnt, 
                                            d_segment_ncf_spectrum_batch, nfft, 
                                            nspec, batch_data_unit_count);

        csum2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_spectrum_batch, nfft, d_segment_ncf_spectrum_batch, nfft, nspec, d_proccnt, nstep);
      }

      // ----------------------- Hilbert Transform --------------------------------
      // CUDACHECK(cudaMemcpy(hilbert_buffer + (all_finishcnt + d_finishcnt) * nfft, d_total_ncf_spectrum_batch, d_proccnt * nfft * sizeof(cuComplex), cudaMemcpyDeviceToDevice));
       
      // ----------------------- Hilbert Transform --------------------------------

      cufftExecC2R(plan, (cufftComplex *)d_total_ncf_spectrum_batch, (cufftReal *)d_total_ncf_batch);
      DimCompute(&dimgrd, &dimblk, nfft, d_proccnt);
      InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_total_ncf_batch, nfft, nfft, d_proccnt, delta);
      CUDACHECK(cudaMemcpy(ncf_buffer + (all_finishcnt + d_finishcnt) * nfft, d_total_ncf_batch, d_proccnt * nfft * sizeof(float), cudaMemcpyDeviceToHost));
      
      CUDACHECK(cudaMemcpy2D(d_ncf_buffer_complex_d + (all_finishcnt + d_finishcnt) * nfft, 2 * sizeof(float), d_total_ncf_batch, 1 * sizeof(float), sizeof(float), d_proccnt * nfft, cudaMemcpyDeviceToDevice));

      // NOTE: here cuda_calc finished
      for(size_t i = 0; i < d_proccnt; i++) {
        SHAREDITEM *ptr = pItem + globalidx_batch;
        pthread_mutex_lock(&(ptr->mtx));
        if (ptr->valid == -1) {
          ptr->phead = &((pPairList + globalidx_batch)->headncf);
          ptr->pdata = ncf_buffer + (all_finishcnt + d_finishcnt + i) * nfft + nspec - nhalfcc - 1;
          char *ncf_fileName = GetEachNcfPath(pSpecSrcList[(pPairList + globalidx_batch)->srcidx].filepath,
                                  pSpecStaList[(pPairList + globalidx_batch)->staidx].filepath,
                                  ncf_dir);
          strcpy(ptr->fname, ncf_fileName);
          
          ptr->valid = 0;
        }
        pthread_mutex_unlock(&(ptr->mtx));
        globalidx_batch++;
      }

      d_finishcnt += d_proccnt;
    }

    // tag: xc_end_time
    clock_gettime(CLOCK_MONOTONIC, &xc_end_time);
    double elapsed_xc_time = getElapsedTime(xc_start_time, xc_end_time);
    xc_time[gpu_batch] = elapsed_xc_time;
    

    all_finishcnt += current_batch_size;
  }

  size_t gpu_avail_ram_before_xc = QueryAvailGpuRam(gpu_id);
  GpuFree((void **)&d_src_spectrum_batch);
  GpuFree((void **)&d_sta_spectrum_batch);
  size_t gpu_avail_ram_after_xc = QueryAvailGpuRam(gpu_id);
  // ----------------------------- PWS Stack --------------------------------
  struct timespec start_pws_stack_time, end_pws_stack_time;
  clock_gettime(CLOCK_MONOTONIC, &start_pws_stack_time);

  cuComplex *ncf_buffer_complex_halfcc;
  GpuMalloc((void **)&ncf_buffer_complex_halfcc, paircnt * npts * sizeof(cuComplex));
  size_t dst_pitch = npts * sizeof(cuComplex);
  size_t src_pitch = nfft * sizeof(float);
  size_t width = npts * sizeof(cuComplex);
  size_t height = paircnt;
  CUDACHECK(cudaMemcpy2D(ncf_buffer_complex_halfcc, dst_pitch, d_ncf_buffer_complex_d + (nspec - nhalfcc - 1), src_pitch, width, height, cudaMemcpyDeviceToDevice));

  cufftHandle c2c_plan_npts;
  cufftPlanMany(&c2c_plan_npts, 1, &npts, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, paircnt);

  struct timespec start_hilbert_time, end_hilbert_time;
  clock_gettime(CLOCK_MONOTONIC, &start_hilbert_time);
  HibertTransform(ncf_buffer_complex_halfcc, npts, paircnt, G_SIZE, B_SIZE, c2c_plan_npts);
  clock_gettime(CLOCK_MONOTONIC, &end_hilbert_time);
  double elapsed_hilbert_time = getElapsedTime(start_hilbert_time, end_hilbert_time);

  float *d_mean_amp_npts, *d_weighted_amp_npts;
  GpuMalloc((void **)&d_mean_amp_npts, npts * sizeof(float));
  GpuMalloc((void **)&d_weighted_amp_npts, npts * sizeof(float));
  struct timespec start_pws_time, end_pws_time;
  clock_gettime(CLOCK_MONOTONIC, &start_pws_time);
  cudaPwsStack(ncf_buffer_complex_halfcc, npts, paircnt, d_mean_amp_npts, d_weighted_amp_npts, G_SIZE, B_SIZE);
  clock_gettime(CLOCK_MONOTONIC, &end_pws_time);
  double elapsed_pws_time = getElapsedTime(start_pws_time, end_pws_time);

  float *h_mean_amp_npts, *h_weighted_amp_npts;
  CpuMalloc((void **)&h_mean_amp_npts, npts * sizeof(float));
  CpuMalloc((void **)&h_weighted_amp_npts, npts * sizeof(float));
  CUDACHECK(cudaMemcpy(h_mean_amp_npts, d_mean_amp_npts, npts * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_weighted_amp_npts, d_weighted_amp_npts, npts * sizeof(float), cudaMemcpyDeviceToHost));

  /*
  // Print_Complex<<<1,1>>>(hilbert_buffer, nfft * paircnt);
  cufftPlanMany(&c2c_plan, 1, &nfft, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, paircnt);
  HibertTransform_Spectrum(hilbert_buffer, nfft, paircnt, G_SIZE, B_SIZE, c2c_plan);
  cudaDeviceSynchronize();
  // Print_Complex<<<1,1>>>(hilbert_buffer, nfft * paircnt);

  float *d_mean_amp, *d_weighted_amp;
  GpuMalloc((void **)&d_mean_amp, nfft * sizeof(float));
  GpuMalloc((void **)&d_weighted_amp, nfft * sizeof(float));
  
  cudaPwsStack(hilbert_buffer, nfft, paircnt, d_mean_amp, d_weighted_amp, G_SIZE, B_SIZE);

  float *h_mean_amp, *h_weighted_amp;
  CpuMalloc((void **)&h_mean_amp, nfft * sizeof(float));
  CpuMalloc((void **)&h_weighted_amp, nfft * sizeof(float));
  CUDACHECK(cudaMemcpy(h_mean_amp, d_mean_amp, nfft * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_weighted_amp, d_weighted_amp, nfft * sizeof(float), cudaMemcpyDeviceToHost));

  clock_gettime(CLOCK_MONOTONIC, &end_pws_stack_time);
  double elapsed_pws_stack_time = getElapsedTime(start_pws_stack_time, end_pws_stack_time);

  float *h_mean_amp_halfcc, *h_weighted_amp_halfcc;
  CpuMalloc((void **)&h_mean_amp_halfcc, npts * sizeof(float));
  CpuMalloc((void **)&h_weighted_amp_halfcc, npts * sizeof(float));
  CUDACHECK(cudaMemcpy(h_mean_amp_halfcc, d_mean_amp + nspec - nhalfcc - 1, npts * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_weighted_amp_halfcc, d_weighted_amp + nspec - nhalfcc - 1, npts * sizeof(float), cudaMemcpyDeviceToHost));
  */

  clock_gettime(CLOCK_MONOTONIC, &end_pws_stack_time);
  double elapsed_pws_stack_time = getElapsedTime(start_pws_stack_time, end_pws_stack_time);

  char *out_pws_weighted_sac = createPwsWeightedFilePath(stack_dir, sta_pair_copy, base_name_copy);
  char *out_pws_mean_sac = createPwsMeanFilePath(stack_dir, sta_pair_copy, base_name_copy);

  // char *out_pws_sac = "/home/woodwood/hpc/station_2/z-test/stack_test_2year_sort/pws-sac/AAKH-ABNH.U-U.pws.sac";
  // char *out_pws_mean_sac = "/home/woodwood/hpc/station_2/z-test/stack_test_2year_sort/pws-sac/AAKH-ABNH.U-U.pws_mean.sac";
  printf("[INFO]: out_pws_weighted_sac: %s\n", out_pws_weighted_sac);
  printf("[INFO]: out_pws_mean_sac: %s\n", out_pws_mean_sac);
  if (create_parent_dir(out_pws_weighted_sac) == -1 || create_parent_dir(out_pws_mean_sac) == -1)
  {
    fprintf(stderr, "Error creating directory %s: ", dirname(out_pws_weighted_sac));
    perror(NULL);
    free(out_pws_weighted_sac);
    free(out_pws_mean_sac);
    return 1;
  }
  write_sac(out_pws_weighted_sac, hdstack, h_weighted_amp_npts);
  write_sac(out_pws_mean_sac, hdstack, h_mean_amp_npts);
  
  // ----------------------------- PWS Stack --------------------------------

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
  double elapsed_stack_time = getElapsedTime(start_stack_time, end_stack_time);
  
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

  // ---------------------------------Start Write NCF file--------------------------------------------
  // tag: write pItem into a file
  if(ncf_dir != NULL) {
    struct timespec start_write_time, end_write_time;
    clock_gettime(CLOCK_MONOTONIC, &start_write_time);
    printf("[INFO]: ---------------Start Writing NCF file------------------\n");
    // const char *all_ncf_sac = "/home/woodwood/hpc/station_2/z-test/stack_test_2year/ncf/AAKH-ABNH.U-U.ncf.sac";
    if(create_parent_dir(ncf_filepath) == -1) {
      fprintf(stderr, "Error creating NCF directory %s: ", dirname(ncf_filepath));
      perror(NULL);
      free(ncf_filepath);
      return 1;
    }
    write_multiple_sac(ncf_filepath, pItem, paircnt);
    clock_gettime(CLOCK_MONOTONIC, &end_write_time);
    double elapsed_write_time = getElapsedTime(start_write_time, end_write_time);
    printf("[INFO]: Write time: %.6f seconds\n", elapsed_write_time);
    printf("[INFO]: ---------------Finish Writing NCF file-----------------\n");
  }
  
  // ---------------------------------End Write NCF file--------------------------------------------

  // tag: endtime
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double elapsed_time = getElapsedTime(start_time, end_time);

  // tag: free mem time
  struct timespec start_free_time, end_free_time;
  clock_gettime(CLOCK_MONOTONIC, &start_free_time);

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

  // Free gpu memory for each batch
  GpuFree((void **)&d_pairlist_batch);
  GpuFree((void **)&d_segment_ncf_spectrum_batch);
  GpuFree((void **)&d_total_ncf_spectrum_batch);
  GpuFree((void **)&d_total_ncf_batch);

  // ----------------- Free gpu memory for pws_stack process -----------------
  GpuFree((void **)&ncf_buffer_complex_halfcc);
  GpuFree((void **)&d_mean_amp_npts);
  GpuFree((void **)&d_weighted_amp_npts);

  CpuFree((void **)&h_mean_amp_npts);
  CpuFree((void **)&h_weighted_amp_npts);

  // CpuFree((void **)&h_mean_amp);
  // CpuFree((void **)&h_weighted_amp);

  CUFFTCHECK(cufftDestroy(plan));
  freeFilePaths(pSrcPaths);
  freeFilePaths(pStaPaths);

  clock_gettime(CLOCK_MONOTONIC, &end_free_time);
  double elapsed_free_time = getElapsedTime(start_free_time, end_free_time);

  // ----------------- Free gpu memory for pws_stack process -----------------

  printf("[INFO]: -----------------------------End of Program-----------------------------------------\n");
  printf("[INFO]: Read File Toal time: %.6f seconds\n", elapsed_readFile_time);

  printf("[INFO]: Read Merge File time: %.6f seconds\n", elapsed_read_MergeFile_time);

  printf("[INFO]: Gen Spec Array time: %.6f seconds\n", elapsed_genSpecArray_time);

  printf("[INFO]: Gen Pair and Pre-process time: %.6f seconds\n", elapsed_genpair_time);

  printf("[INFO]: Gpu Alloc time: %.6f seconds\n", elapsed_gpu_alloc_time);

  double xc_time_sum = 0;
  for(int i = 0; i < total_batches; i++) {
    printf("[INFO]: XC time for batch %d: %.6f seconds\n", i, xc_time[i]);
    xc_time_sum += xc_time[i];
  }
  printf("[INFO]: XC time: %.6f seconds\n", xc_time_sum);

  printf("[INFO]: PWS Stack time: %.6f seconds\n", elapsed_pws_stack_time);

  // printf("[INFO]: cpy1 time: %.6f seconds\n", elapsed_cpy1_time);

  // printf("[INFO]: Hilbert Transform time: %.6f seconds\n", elapsed_hilbert_time);

  // printf("[INFO]: pws time: %.6f seconds\n", elapsed_pws_time);

  printf("[INFO]: Linear Stack time: %.6f seconds\n", elapsed_stack_time);

  printf("[INFO]: Free Mem time: %.6f seconds\n", elapsed_free_time);

  printf("[INFO]: Total time: %.6f seconds\n", elapsed_time);

  return 0;
}
