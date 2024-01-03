### 1. pthread_mutex_t

```cpp
typedef struct
{
  pthread_mutex_t mtx;
  int valid; /* -1: default; 1: ready to file; 2: finish to file */
  char fname[PATH_MAX];
  SACHEAD *phead;
  float *pdata;
} SHAREDITEM;
```
在线程实际运行过程中，经常需要多个线程保持同步。这时可以用互斥锁来完成任务；互斥锁的使用过程中，主要有`pthread_mutex_init`，`pthread_mutex_destory`，`pthread_mutex_lock`，`pthread_mutex_unlock`这几个函数以完成锁的初始化，锁的销毁，上锁和释放锁操作。

https://www.cnblogs.com/hyacinthLJP/p/16795204.html

 
### 2. timespec

```cpp
在Linux中常用的时间结构有struct timespec 和struct timeval 。
下面是两个结构的定义
struct timespec
{
  __time_t tv_sec;        /* Seconds. */
  long   tv_nsec;       /* Nanoseconds. */
};
struct timeval {
　time_t tv_sec;  
　suseconds_t tv_usec;
}; 
```
两者的区别是timespec的第二个参数是纳秒数，而timeval的第二个参数是毫秒数。

### 3. get npts of ouput NCF from -cclength to cclength

```cpp
  int nhalfcc = floorf((cclength / delta) + 1e-7);
  int ncc = 2 * nhalfcc + 1;
  printf("[INFO]: cclength = %f\n", cclength);
  printf("[INFO]: delta = %f\n", delta);
  printf("[INFO]: ncc = %d\n", ncc);
```
这段代码的目的是计算输出 NCF（Cross-Correlation Function）的点数，各个参数的含义：

1. `cclength`：
  类型：float 
  含义：这是交叉相关函数的长度，即在时间上的范围。它表示你希望计算的交叉相关函数的时间跨度。

2. `delta`：
  类型：float
  含义：这是采样的时间间隔。它表示每两个相邻数据点之间的时间间隔。

3. `nhalfcc`：
  类型：int
  含义：这是 cclength 在时间单位上的半长度。计算方式是将 cclength 除以 delta，然后取 floor（向下取整）。这样计算得到的 nhalfcc 表示了在时间上的半窗口宽度，用于限制交叉相关函数的计算范围。

4. `ncc`：
  类型：int
  含义：这是最终用于输出 NCF 的数据点数。它是 nhalfcc 的两倍再加一，以确保覆盖从 -cclength 到 cclength 的完整时间范围。

### 4. 整个函数的执行流程

1. 解析命令行参数：包括光谱长度`cclength`、NCF目录`ncf_dir`、交叉相关性`xcorr`、GPU ID`gpu_id`等。

2. 设置GPU设备：使用`cudaSetDevice`函数设置当前设备为选定的GPU。
  ```cpp
  CUDACHECK(cudaSetDevice(gpu_id));
  ```

3. 读取输入文件路径列表：使用`read_filelist`函数读取输入文件路径列表。
  ```cpp
  FilePaths *SpecPaths = read_spec_lst(parg->spectrum_lst);
```
4. 预定义和解析：然后，进行一系列的预定义和解析操作，包括读取光谱头部信息，计算NCF的点数，打印相关信息等。

5. CPU内存分配：接着，为光谱节点、对节点、光谱数据缓冲区等分配CPU内存。

6. 数据读取：然后，从文件路径列表中读取数据到光谱节点的数据缓冲区。

7. 对生成：接着，生成对列表，并为NCF缓冲区分配CPU内存。

8. 头部处理：然后，设置输出NCF的头部信息。
  ```cpp
  // Set the head of output NCF
  for (size_t i = 0; i < paircnt; i++)
  {
    SACHEAD *phdncf = &(ppairlist[i].headncf);
    SEGSPEC *phd_src = &(pspeclist[ppairlist[i].srcidx].head);
    SEGSPEC *phd_sta = &(pspeclist[ppairlist[i].staidx].head);
    SacheadProcess(phdncf, phd_src, phd_sta, delta, ncc, cclength);
  }
  ```

9. 线程属性设置：接着，初始化共享项目的互斥锁和有效标志。
  ```cpp
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
  ```

10. GPU内存分配：然后，为光谱节点数据缓冲区分配GPU内存，并将数据从CPU缓冲区复制到GPU。
  ```cpp
  // Allocate GPU memory for spectrum node data buffer for input
  GpuMalloc((void **)&d_spectrum, spec_cnt * vec_size);

  // Copy spectrum data from CPU buffer to GPU
  CUDACHECK(cudaMemcpy(d_spectrum, spectrum_buffer, spec_cnt * vec_size, cudaMemcpyHostToDevice));

  ```

11. GPU动态内存分配：接着，根据批处理大小，为对节点、NCF数据等分配GPU动态内存。
  ```cpp
  // Alloc gpu dynamic memory with d_batch
  CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, d_batch);
  GpuMalloc((void **)&d_pairlist, d_batch * sizeof(PAIRNODE));
  GpuMalloc((void **)&d_segment_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf_spectrum, d_batch * nfft * sizeof(complex));
  GpuMalloc((void **)&d_total_ncf, d_batch * nfft * sizeof(float));
  ```

12. 交叉相关计算：然后，进行交叉相关计算，包括GPU处理、数据复制、路径生成等。
  ```cpp
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
  ```

13. 释放内存：最后，释放所有的内存。

### 5. h_batch 批处理
批处理是一种常见的数据处理技术，主要用于处理大量数据。其基本思想是将大量的数据分成多个小批次（batch），然后逐个批次进行处理。这种方法可以有效地管理内存资源，提高计算效率，特别是在使用 GPU 进行并行计算时。

```cpp
  // reading data from [SpecPaths] to [pspeclist.pdata]
  GenSpecArray(SpecPaths, pspeclist); // Find something error
  size_t paircnt = GeneratePair(ppairlist, pspeclist, spec_cnt, xcorr);
  h_batch = (h_batch > paircnt) ? paircnt : h_batch;
  // Alloc CPU output memory
  CpuMalloc((void **)&ncf_buffer, h_batch * nfft * sizeof(float));
```
在这段代码中，h_batch 变量表示每个批次的大小。在处理光谱数据时，首先生成所有可能的对列表，然后将这些对分成多个批次，每个批次包含 h_batch 个对。然后，每个批次的数据被加载到 GPU 中进行处理。

批处理的主要优点如下：

1. 内存管理：当处理大量数据时，可能会超出内存容量。通过批处理，可以将数据分成多个小批次，每个批次的大小可以根据可用内存来设定，从而避免内存溢出。

2. 并行计算：在 GPU 计算中，可以同时处理一个批次中的所有数据，从而实现并行计算，提高计算速度。

3. 网络训练：在深度学习中，批处理还可以用于网络训练。通过在每个批次上计算梯度并更新网络参数，可以实现随机梯度下降（SGD）或其变体，从而提高训练效率。

4. 硬件优化：批处理可以更好地利用硬件资源，如 CPU 的缓存和 GPU 的内存带宽，从而提高计算效率。