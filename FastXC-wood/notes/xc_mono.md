# 代码注释

## 一、xc-0.9.0

###  xc_mono

#### 1. `arguproc`: 用于解析命令行参数
`ARGUTYPE`结构体定义如下：
```cpp
typedef struct ARGUTYPE
{
  char *spectrum_lst; // dir of segspec
  char *ncf_dir;
  float cclength; /* half length of output NCF */
  int gpu_id;     /* GPU ID */
  int xcorr;      /* If do cross-correlation */
} ARGUTYPE;
```
*****
 
#### 2. `cal_dist`：计算地球上两点之间的距离和方位角
```cpp
void distkm_az_baz_Rudoe(double evlo,double evla,double stlo,double stla,double *gcarc,double *az,double* baz,double *distkm)
```
其中各个参数的含义为：
* `double evlo`, `double evla`：事件的经度和纬度, 可能是地震事件或其他地理事件的位置。
* `double stlo`, `double stla`：站点的经度和纬度。这可能是地震监测站或其他观测站的位置。
* `double *gcarc`：大圆弧距离的指针。这是两点之间的球面距离，通常以度为单位。
* `double *az`：方位角的指针。这是从站点到事件的方向，通常以度为单位，从北向顺时针测量。
* `double *baz`：反方位角的指针。这是从事件到站点的方向，通常以度为单位，从北向顺时针测量。
* `double *distkm`：距离的指针，单位是千米。函数会计算两点之间的距离，并通过这个指针返回。

地球半径，地球扁率，第一偏心率，第二偏心率
```cpp
    /* earth constant of WGS84. note it is a little different from sac  */
    double EARTHR  = 6378.137;              /* Earth Radius,major axis,eg radius at the equator */
    double EARTHFL = 1.0/298.257223563 ;    /* Earth Flattening factor,fl=(a-b)/a */
    /* ellipsoid constant for earth. When convert geographical latitude into geocentrical latitude,
     *  use the formula tan(geoCentLat)=ONEMEC2*tan(geoGraphLat) */
    double EC2,ONEMEC2;
    EC2     = 2.0 * EARTHFL - EARTHFL * EARTHFL;   /* EC2=e^2=(a^2-b^2)/a^2 */
    ONEMEC2 = 1.0 - EC2;                           /* ONEMEC2=one minus EC2=b^2/a^2 */
#ifdef nouse
    double EPS;
    EPS     = 1.0 + EC2/ONEMEC2;                   /* EPS=a^2/b^2 >1 */
#endif
```
*****


#### 3. `complex`:定义了一个名为 complex 的结构体
有两个 float 类型的成员：x 和 y, 用于表示复数
```cpp
typedef struct
{
    float x;
    float y;
} complex;
```
*****

#### 4. `config`:一些常量
如`MAXLINE=8192`, `MAXPATH=8192`, `MAXNME=255`

*****

#### 5. `cuda.main.cu`



*****

#### 6. `cuda.util.cu`
`const float RAMUPPERBOUND = 0.9;` 限制 GPU 内存的使用量为 0.9
1. `DimCompute`
    ```cpp
    void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
    ```
    这个函数用于计算 CUDA 核函数的线程块（block）和网格（grid）的维度。
    在 CUDA 中，`dim3`是一种用于描述线程块和线程网格维度的数据类型。GPU 上的任务通常被组织成线程块（blocks）和线程网格（grids）。每个线程块包含若干个线程，而线程块本身则被组织成线程网格。
    <br>
  

2. `QueryAvailGpuRam`
    ```cpp
    size_t QueryAvailGpuRam(size_t deviceID)
    ```
    这个函数用于查询指定 GPU 设备上可用 GPU 内存。
    <br>

3. `EstimateGpuBatch`
    ```cpp
    size_t EstimateGpuBatch(size_t gpu_id, size_t fixedRam, size_t unitram,
                            int numType, int rank, int *n, int *inembed,
                            int istride, int idist, int *onembed, int ostride,
                            int odist, cufftType *typeArr)
    ```
    这个函数用于估计在给定的 GPU 内存限制下，可以处理的最大批次大小。批处理大小是指在一次调用中，GPU 可以处理的数据量。这个函数的输入参数包括：
    * `size_t gpu_id`：GPU 设备 ID；
    * `size_t fixedRam`：GPU 内存的固定使用量；
    * `size_t unitram`：每批次的内存需求；
    * `int numType`：FFT 类型的数量；
    * `int rank`：FFT 的秩；
    * `int *n`：FFT 的维度；
    * `int *inembed`：输入嵌入数组；
    * `int istride`：输入跨度；
    * `int idist`：输入距离；
    * `int *onembed`：输出嵌入数组；
    * `int ostride`：输出跨度；
    * `int odist`：输出距离；
    * `cufftType *typeArr`：包含 FFT 类型的数组。

    `cufftType` 是 `CUDA Fast Fourier Transform` 库（cuFFT）中用于表示 FFT 类型的枚举类型。cuFFT 库是NVIDIA 提供的用于在 GPU 上执行快速傅里叶变换（FFT）的库。`cufftType` 枚举类型定义了不同的 FFT 类型，用于指定 FFT 操作的精度和方向。

    枚举类型 cufftType 的可能取值包括：

    * `CUFFT_R2C`：实数到复数的 FFT（实部到复部）。
    * `CUFFT_C2R`：复数到实数的 FFT（复部到实部）。
    * `CUFFT_C2C`：复数到复数的 FFT。
    * `CUFFT_D2Z`：双精度实数到双精度复数的 FFT（实部到复部）。
    * `CUFFT_Z2D`：双精度复数到双精度实数的 FFT（复部到实部）。
    * `CUFFT_Z2Z`：双精度复数到双精度复数的 FFT。

    这个函数首先调用 `QueryAvailGpuRam` 函数，查询指定 GPU 的可用内存。检查固定内存需求是否超过了可用内存。如果超过了，就打印错误信息并退出程序。接着，通过循环调用 `cufftEstimateMany` 函数，计算 FFT 的内存需求。这个函数会根据 FFT 的参数，估计执行 FFT 所需的内存大小。不断增加批次大小，直到总内存需求超过了可用内存。在每次循环中，都会增加批次大小，并更新总内存需求。
    <br>

4. `CufftPlanAlloc`: 封装了 cuFFT 计划的创建过程
    ```cpp
    void CufftPlanAlloc(cufftHandle *pHandle, int rank, int *n, int *inembed,
                        int istride, int idist, int *onembed, int ostride,
                        int odist, cufftType type, int batch)
    {
    // create cufft plan
    CUFFTCHECK(cufftPlanMany(pHandle, rank, n, inembed, istride, idist, onembed,
                            ostride, odist, type, batch));
    }
    ```
    `cufftHandle *pHandle`: 指向 `cufftHandle` 类型变量的指针 `pHandle`
    `CUFFTCHECK` 是一个宏，用于检查 cuFFT 函数的返回值。如果 `cufftPlanMany` 函数执行成功，它会返回 CUFFT_SUCCESS。如果返回其他值，表示发生了错误，`CUFFTCHECK` 宏会打印错误信息并退出程序。
    <br>

5. `GpuMalloc`: 封装了 GPU 内存分配函数 `cudaMalloc`
    ```cpp
    void GpuMalloc(void **pptr, size_t sz) { CUDACHECK(cudaMalloc(pptr, sz)); }
    ```
    <br>

6. `GpuCalloc`: 封装了 GPU 内存分配函数 `cudaMalloc` 和 `cudaMemset`
    ```cpp
    void GpuCalloc(void **pptr, size_t sz)
    {
    CUDACHECK(cudaMalloc(pptr, sz));

    CUDACHECK(cudaMemset(*pptr, 0, sz));
    }
    ```
    <br>

7. `GpuFree`: 封装了 GPU 内存释放函数 `cudaFree`
    ```cpp
    void GpuFree(void **pptr)
    {
    CUDACHECK(cudaFree(*pptr));
    *pptr = NULL;
    }
    ```
*****

#### 7. `cuda.xc_mono.cu`

1. `cmulmono2DKernel`: 用于在 GPU 上执行复数乘法操作，并将结果存储到 d_segncf 中。
    参数：
    * `cuComplex *d_spec`: 指向 `cuComplex` 类型的指针 `d_spec`
    * `size_t srcpitch, size_t srcoffset`: 源数据的pitch 和 offset
    * `size_t stapitch, size_t staoffset`: 站点数据的 pitch 和 offset
    * `PAIRNODE *d_pairlist`: 指向 `PAIRNODE` 类型的指针 `d_pairlist`
    * `size_t paircnt`: 交叉相关对的数量
    * `cuComplex *d_segncf`: 指向 `cuComplex` 类型的指针 `d_segncf`, 用于存储分段 NCF 数据，是用于存储归一化互相关函数结果的数组。
    * `size_t ncfpitch`: NCF 数据的 pitch
    * `int nspec`: 谱线数量 nspec

    `ncf`: noise cross correlation
<br>

2. `sum2DKernel`: 用于在 GPU 对复数数组进行累计求和 cumsum 操作
    ```cpp
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    ```
    这里，通过计算索引 `sidx` 和 `didx`，确定线程在二维数组中的位置。这里使用了 `spitch` 和 `dpitch`，它们是输入和输出数组的行偏移。
    ```cpp
    cuComplex temp = d_segment_spectrum[sidx];
    temp.x /= nstep; // divide the real part by nstep
    temp.y /= nstep; // divide the imaginary part by nstep
    ```
    从输入数组 `d_segment_spectrum` 中获取复数值，并将其实部和虚部分别除以 `nstep`。这个操作可能是为了计算平均值。
    ```cpp
    d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
    ```
    在有效范围内的线程，将复数数组 `d_total_spectrum` 中的相应位置进行操作，将计算得到的复数值加到输出数组上，以实现对复数数组的求和操作。`cuCaddf` 是 `cuComplex` 库中用于复数相加的函数。
<br>

3. `InvNormalize2DKernel`: 用于在 GPU 上执行归一化操作
    其中计算了一个权重值，用于归一化操作。这个权重是根据数组的宽度 width 和一个给定的参数 dt 计算得到的。
    ```cpp
    double weight = 1.0 / (width * dt);
    ```

*****


#### 8. `gen_ccfpath`
1. `CreateDir`: 创建一个目录路径。如果路径中的任何一个目录不存在，它都会尝试创建
<br>

2. `SplitFileName`: 于将文件名分割成多个部分, 存储分割结果的字符指针 `stastr`、`yearstr`、`jdaystr`、`hmstr` 和 `chnstr`
<br>

3. `SacheadProcess`: 用于处理 SAC (Seismic Analysis Code) 文件的头部信息。
这个函数接受六个参数：一个指向 `SACHEAD` 结构体的指针 `ncfhd`，两个指向 `SEGSPEC` 结构体的指针 `srchd` 和 `stahd`，一个浮点数 `delta`，一个整数 `ncc`，以及一个浮点数 `cclength`。
    * `delta` 是一个浮点数，表示采样间隔。在地震学中，采样间隔是指连续两个地震信号采样点之间的时间间隔，通常以秒为单位。

    * `ncc` 是一个整数，表示数据点数。在这个函数中，ncc 被用来设置 `ncfhd->npts`，即 SAC 文件头部的数据点数。

    * `cclength` 是一个浮点数，表示交叉相关函数（Cross-Correlation Function）的长度。在这个函数中，`cclength` 被用来设置 `ncfhd->b` 和 `ncfhd->e`，即 SAC 文件头部的开始时间和结束时间。开始时间被设置为 `-1.0 * cclength`，结束时间被设置为 `cclength`，这意味着交叉相关函数是以 0 为中心的。
<br>

4. `GenCCFPath`: 根据源文件和台站文件的路径，生成一个交叉相关的输出路径。这个路径包含了从源文件和台站文件的路径中提取的信息，包括台站名、年份、日数、小时分钟和通道名。
这个函数接受四个参数：一个字符指针 `ccf_path`，用于存储生成的路径；两个字符指针 `src_path` 和 `sta_path`，分别表示源文件和台站文件的路径；以及一个字符指针 `output_dir`，表示输出目录。

*****

#### 9. `gen_pair_mono`
1. `cmpspec`:比较两个 SEGSPEC 结构体的内容是否相等，如果任何一个字段不相等，那么函数就返回 -1。
    ```cpp
    typedef struct segspec_s
    {
    float stla; // 纬度
    float stlo; // 经度
    /* segment info */
    int nstep;  // 地震信号的段数
    
    /* FFT info  */
    int nspec; /* use fftr() number of complex eg 2*nspec float */
    float df;
    float dt;

    } SEGSPEC;
    ```
    
    * `nspec`: FFT（快速傅里叶变换）的结果中的复数个数。在 FFT 中，输入信号的长度通常需要是 2 的幂，输出结果是一系列的复数，每个复数都对应输入信号的一个频率成分。`nspec` 的值通常是输入信号长度的一半，因为 FFT 的结果是对称的。

    * `df`: 是一个浮点数，表示频率的间隔。在 FFT 中，输出结果的每个复数都对应一个频率，这些频率是等间隔的，间隔的大小就是 `df`。

    * `dt`: 是一个浮点数，表示采样间隔。在地震学中，采样间隔是指连续两个地震信号采样点之间的时间间隔，通常以秒为单位。

<br>

2. `GenSpecArray`: 根据 `fileList` 中的文件路径生成一个 `SPECNODE` 结构体数组。
    ```cpp
    typedef struct SPECNODE
    {
        int valid;
        char filepath[MAXLINE];
        SEGSPEC head;
        complex *pdata;
    } SPECNODE;
    ```
    每个 `SPECNODE` 结构体都包含一个文件路径、一个头部信息和一些数据。如果一个文件的内容成功被读取，那么对应的 `SPECNODE` 就被标记为有效。

<br>

3. `extractFileInfo`: 从文件名中提取地震数据文件的信息，包括台站名称、年份、儒略日、小时、分钟和`component`。
    ```cpp
    typedef struct
    {
        char *station;
        int year;
        int julianday;
        int hour;
        int minute;
        char *component;
    } FileInfo;
    ```

<br>

4. `GeneratePair`: 生成一组 PAIRNODE 结构体。
    ```cpp
    typedef struct PAIRNODE
    {
        size_t srcidx;
        size_t staidx;
        SACHEAD headncf;
    } PAIRNODE;
    ```
    这个函数接受四个参数：一个 `PAIRNODE` 结构体指针 `ppairlist`，一个 `SPECNODE` 结构体指针 `plist`，一个 `size_t` 类型的 `spec_cnt` 和一个整数 `xcorr`。
    该函数使用两个嵌套的 `for` 循环遍历 `plist` 中的所有 `SPECNODE` 结构体，生成所有可能的对，并将这些对存储到 `ppairlist` 中。如果 `xcorr` 为 0，那么就跳出内部的 `for` 循环，只进行自相关计算。
    函数最后返回生成的 `PAIRNODE` 结构体的数量。

*****

#### 10. `node_util`
```cpp
typedef struct FilePaths
{
    char **paths;
    int count;
} FilePaths;

typedef struct SPECNODE
{
    int valid;
    char filepath[MAXLINE];
    SEGSPEC head;
    complex *pdata;
} SPECNODE;

typedef struct PAIRNODE
{
    size_t srcidx;
    size_t staidx;
    SACHEAD headncf;
} PAIRNODE;
```
*****

#### 11. `read_segspec`
1. `read_spec_buffer`: 用于从文件中读取地震频谱数据，并将数据存储在预先分配的缓冲区中。函数执行成功，那么返回的就是传入的 `buffer` 指针。如果函数在任何一个步骤中失败，那么返回的就是 `NULL`。

2. `read_spechead`: 用于从文件中读取地震频谱的头部信息。函数尝试从文件中读取 `SEGSPEC` 结构体的大小的数据，并将数据存储在 `hd` 指向的结构体中。如果函数执行成功，那么返回的就是 `0`。如果函数在任何一个步骤中失败，那么返回的就是 `-1`。
*****

#### 12. `read_spec_lst`
1. `read_spec_lst`: 读取一个包含多个文件路径的文件，并将这些路径存储在一个 `FilePaths` 结构体中, 每添加一个路径，`count` 就增加 1。
    ```cpp
    typedef struct FilePaths
    {
        char **paths;
        int count;
    } FilePaths;
    ```

2. `freeFilePaths`: 用于释放 `FilePaths` 结构体的内存, 每次循环中，函数释放 fp->paths[i] 指向的内存。

*****


#### 13. `sac`
定义了 `SACHEAD` 结构体以及一些宏定义
*****


#### 14. `sacio`
* SAC I/O functions:
*	read_sachead	read SAC header
*	read_sac	read SAC binary data
*	read_sac2	read SAC data with cut option
*	write_sac	write SAC binary data
*	swab4		reverse byte order for integer/float
*	wrtsac2		write 2 1D arrays as XY SAC data
*	sachdr		creat new sac header
*	rdsac0_		fortran wraper for read_sac
*	my_brsac_	fortran binary sac data+head reader.
*	wrtsac0_	fortran write 1D array as SAC binary data
*	wrtsac2_	fortran wraper for wrtsac2
*	wrtsac3_	wrtsac0 with component orientation cmpaz/cmpinc
*	ResetSacTime	reset reference time in the head
*	sac_head_index	find the index of a sac head field.
*****


#### 15. `segspec`
```cpp
typedef struct segspec_s
{
  float stla;
  float stlo;
  /* segment info */
  int nstep;

  /* FFT info  */
  int nspec; /* use fftr() number of complex eg 2*nspec float */
  float df;
  float dt;

} SEGSPEC;
```
*****


#### 16. `usage`:打印程序的使用方法
```cpp
void usage()
{
    fprintf(
        stderr,
        "\nUsage:\n"
        "specxc_mg -A spec_lst -O out_dir -C halfCCLength -G gpu_id [-X do cross-correlation] \n"
        "Options:\n"
        "    -A Input spectrum list\n"
        "    -O Output directory for NCF files as sac format\n"
        "    -C Half of cclenth (in seconds).\n"
        "    -G ID of Gpu device to be launched \n"
        "    -X Optional. If set, do cross-correlation; else, only do "
        "auto-correlation.\n"
        "Version:\n"
        "  last update by wangjx@20230627\n"
        "  cuda version\n");
}
```
使用方法：`specxc_mg -A spec_lst -O out_dir -C halfCCLength -G gpu_id [-X do cross-correlation]`。这是程序的命令行格式，其中 -A、-O、-C 和 -G 是必需的选项，-X 是可选的。

选项说明：每个选项的具体含义如下：

* -A：输入光谱列表。
* -O：NCF 文件的输出目录，文件格式为 sac。
* -C：半个互相关长度（以秒为单位）。
* -G：要启动的 GPU 设备的 ID。
* -X：可选。如果设置，执行交叉相关；否则，只执行自相关。
版本信息：包括最后更新日期和 CUDA 版本。
*****


#### 17. `util`
1. `QueryAvailCpuRam`: 用于查询系统中可用的 CPU RAM 的大小

2. `CpuMalloc`: 在 CPU 内存中分配指定大小的空间，并处理可能出现的内存分配失败的情况。

3. `CpuCalloc`: 用于在 CPU 内存中分配指定大小的空间，并将这块空间初始化为零。

4. `CpuFree`: 用于释放 CPU 内存中的空间。

5. `EstimateCpuBatch`: 估计可以在 CPU RAM 中分配的批次数量。