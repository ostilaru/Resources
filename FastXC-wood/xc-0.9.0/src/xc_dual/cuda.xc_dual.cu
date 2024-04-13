#include "cuda.xc_dual.cuh"
#include <stdio.h>

// TODO: this function need to be rewrite
__global__ void cmuldual2DKernel(cuComplex *d_specsrcvec, size_t srcpitch, size_t srcoffset,
                                 cuComplex *d_specstavec, size_t stapitch, size_t staoffset,
                                 PAIRNODE *d_pairlist, size_t paircnt,
                                 cuComplex *d_segncfvec, size_t ncfpitch,
                                 int nspec, size_t batch_data_unit_count)
{
  // get the index of the current thread
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // check if the index is out of bound
  if (col < nspec && row < paircnt)
  {
    // NOTE: pitch is used for the 2D array, offset is used for the 1D array
    size_t idx = row * ncfpitch + col;

    // get the index of the source and station
    size_t srcrow, starow;
    size_t srcidx, staidx;
    srcrow = d_pairlist[row].srcidx;
    starow = d_pairlist[row].staidx;

    srcrow %= batch_data_unit_count;
    starow %= batch_data_unit_count;
    
    srcidx = (srcrow * srcpitch + srcoffset + col);
    staidx = (starow * stapitch + staoffset + col);
    
    // cuComplex src = d_specsrcvec[srcidx];
    // cuComplex sta_conj =
    //     make_cuComplex(d_specstavec[staidx].x, -d_specstavec[staidx].y);
        
    cuComplex sta = d_specstavec[staidx];
    cuComplex src_conj =
        make_cuComplex(d_specsrcvec[srcidx].x, -d_specsrcvec[srcidx].y);

    if (col == 0)
    {
      d_segncfvec[idx] = make_cuComplex(0, 0);
    }
    else
    {
      // cuComplex mul_result = cuCmulf(src, sta_conj);
      cuComplex mul_result = cuCmulf(src_conj, sta);
      int sign = (col % 2 == 0) ? 1 : -1;
      d_segncfvec[idx].x = sign * mul_result.x;
      d_segncfvec[idx].y = sign * mul_result.y;
    }
  }
}

// sum2dKernel is used to sum the 2D array of float, not used in the current version
__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height,
                            int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    d_finalccvec[didx] += (d_segncfvec[sidx] / nstep);
  }
}

// TODO: this function need to be rewrite
__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    cuComplex temp = d_segment_spectrum[sidx];
    temp.x /= nstep; // divide the real part by nstep
    temp.y /= nstep; // divide the imaginary part by nstep

    d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
  }
}

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  double weight = 1.0 / (width * dt);
  if (row < height && col < width)
  {
    size_t idx = row * pitch + col;
    d_segdata[idx] *= weight;
  }
}

// -------------------- Hilbert Transform --------------------
__global__ void Divide(cufftComplex *dev_complex, const int LENGTH, const int num_signals)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < LENGTH * num_signals)
    {
        int signal_idx = tid / LENGTH;
        int data_idx = tid % LENGTH;
        dev_complex[tid].x /= LENGTH;
        dev_complex[tid].y /= LENGTH;
    }
}

__global__ void Hilbert(cufftComplex *dev_complex, const int CPX_LEN, const int LENGTH, const int num_signals)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x + 1, // Skips the first element (DC component)
        offset = gridDim.x * blockDim.x;

    if (tid < LENGTH * num_signals)
    {
        int signal_idx = tid / LENGTH;
        int data_idx = tid % LENGTH;
        if (data_idx != 0 && data_idx < CPX_LEN - 1)
        {
            dev_complex[tid].x *= 2;
            dev_complex[tid].y *= 2;
        }
        else if (data_idx > CPX_LEN - 1)
        {
            dev_complex[tid].x = 0.0f;
            dev_complex[tid].y = 0.0f;
        }
    }
}

void HibertTransform(cufftComplex *dev_complex, const int LENGTH, const int num_signals, dim3 G_SIZE, dim3 B_SIZE, cufftHandle c2c_plan)
{
    const int CPX_LEN = LENGTH / 2 + 1;

    cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_FORWARD); // Forward FFT in-place
    CUDACHECK(cudaDeviceSynchronize());

    Hilbert<<<G_SIZE, B_SIZE>>>(dev_complex, CPX_LEN, LENGTH, num_signals);
    CUDACHECK(cudaDeviceSynchronize());

    cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_INVERSE); // Inverse FFT in-place
    CUDACHECK(cudaDeviceSynchronize());

    Divide<<<G_SIZE, B_SIZE>>>(dev_complex, LENGTH, num_signals);
    CUDACHECK(cudaDeviceSynchronize());
}

void HibertTransform_Spectrum(cufftComplex *dev_complex, const int LENGTH, const int num_signals, dim3 G_SIZE, dim3 B_SIZE, cufftHandle c2c_plan)
{
    const int CPX_LEN = LENGTH / 2 + 1;

    // cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_FORWARD); // Forward FFT in-place
    // CUDACHECK(cudaDeviceSynchronize());

    Hilbert<<<G_SIZE, B_SIZE>>>(dev_complex, CPX_LEN, LENGTH, num_signals);
    CUDACHECK(cudaDeviceSynchronize());

    cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_INVERSE); // Inverse FFT in-place
    CUDACHECK(cudaDeviceSynchronize());

    Divide<<<G_SIZE, B_SIZE>>>(dev_complex, LENGTH, num_signals);
    CUDACHECK(cudaDeviceSynchronize());
}

// ------------------- cuda PWS-Stack -------------------
__global__ void cudaMeanReal(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + tid].x;
        }
        // 计算平均值
        mean_amp[tid] = sum_real / num_signals;
    }
}

__global__ void cudaMeanReal_abs(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += abs(hilbert_complex[j * length + tid].x);
        }
        // 计算平均值
        mean_amp[tid] = sum_real / num_signals;
    }
}

__global__ void cudaMeanImg(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_img) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_img = 0.0f;
        // 计算当前时间点上所有信号的虚部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_img += hilbert_complex[j * length + tid].y;
        }
        // 计算平均值
        mean_img[tid] = sum_img / num_signals;
    }
}

__global__ void cudaMean(cufftComplex *hilbert_complex, int length, int num_signals, cufftComplex *mean) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f, sum_img = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + tid].x;
            sum_img += hilbert_complex[j * length + tid].y;
        }
        // 计算平均值
        mean[tid].x = sum_real / num_signals;
        mean[tid].y = sum_img / num_signals;
    }
}

__global__ void cudaABS(cufftComplex *hilbert_complex, int length, int num_signals, float *abs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_signals * length) {
        abs[tid] = sqrt(hilbert_complex[tid].x * hilbert_complex[tid].x + hilbert_complex[tid].y * hilbert_complex[tid].y);
    }
}

__global__ void cudaDivide(cufftComplex *hilbert_complex, int length, int num_signals, float *abs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_signals * length) {
        hilbert_complex[tid].x /= abs[tid];
        hilbert_complex[tid].y /= abs[tid];
    }
}

__global__ void cudaMultiply(float *num1, float *num2, float *res, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        res[tid] = num1[tid] * num2[tid];
    }
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void findMaxValKernel(float *num, float *maxVal, int length) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        atomicMaxFloat(maxVal, num[tid]);
    }
}


__global__ void cudaNormalize(float *num, int length, float *maxVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        num[tid] /= *maxVal;
    }
}

void cudaPwsStack(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp, float *weighted_amp, dim3 G_SIZE, dim3 B_SIZE) {
    // mean_amp: 1024
    cudaMeanReal<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, mean_amp);
    cudaDeviceSynchronize();

    // abs_hilb_amp: 2 * 1024
    float *abs_hilb_amp;
    cudaMalloc(&abs_hilb_amp, num_signals * length * sizeof(float));
    cudaABS<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, abs_hilb_amp);
    cudaDeviceSynchronize();

    // divede_mean: 1024
    cufftComplex *divide, *divide_mean;
    cudaMalloc(&divide, num_signals * length * sizeof(cufftComplex));
    cudaMalloc(&divide_mean, length * sizeof(cufftComplex));
    cudaDivide<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, abs_hilb_amp);
    cudaDeviceSynchronize();
    cudaMean<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, divide_mean);
    cudaDeviceSynchronize();

    // weight: 1024
    float *weight;
    cudaMalloc(&weight, length * sizeof(float));
    cudaABS<<<G_SIZE, B_SIZE>>>(divide_mean, length, 1, weight);
    cudaDeviceSynchronize();

    // weighted_amp: 1024
    cudaMultiply<<<G_SIZE, B_SIZE>>>(mean_amp, weight, weighted_amp, length);
    cudaDeviceSynchronize();

    // normalize mean_amp, weighted_amp
    float *mean_amp_maxVal, *weighted_amp_maxVal;
    cudaMalloc(&mean_amp_maxVal, sizeof(float));
    cudaMalloc(&weighted_amp_maxVal, sizeof(float));
    findMaxValKernel<<<G_SIZE, B_SIZE>>>(mean_amp, mean_amp_maxVal, length);
    findMaxValKernel<<<G_SIZE, B_SIZE>>>(weighted_amp, weighted_amp_maxVal, length);
    cudaDeviceSynchronize();

    cudaNormalize<<<G_SIZE, B_SIZE>>>(mean_amp, length, mean_amp_maxVal);
    cudaNormalize<<<G_SIZE, B_SIZE>>>(weighted_amp, length, weighted_amp_maxVal);
    cudaDeviceSynchronize();

    // free memory
    cudaFree(abs_hilb_amp);
    cudaFree(divide);
    cudaFree(divide_mean);
    cudaFree(weight);
    cudaFree(mean_amp_maxVal);
    cudaFree(weighted_amp_maxVal);

    printf("[INFO]: ------------------------------------------------------\n");

}

// ------------------- host PWS-Stack -------------------
void calculateMeanReal(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + i].x;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_amp[i] = sum_real / num_signals;
    }
}

void calculateMeanReal_abs(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += abs(hilbert_complex[j * length + i].x);
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_amp[i] = sum_real / num_signals;
    }
}

void calculateMeanImg(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_img) {
    for (int i = 0; i < length; ++i) {
        float sum_img = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_img += hilbert_complex[j * length + i].y;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_img[i] = sum_img / num_signals;
    }
}

void calculateMean(cufftComplex *hilbert_complex, size_t length, int num_signals, cufftComplex *mean) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f, sum_img = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + i].x;
            sum_img += hilbert_complex[j * length + i].y;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean[i].x = sum_real / num_signals;
        mean[i].y = sum_img / num_signals;
    }
}

void calculateABS(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs) {
    for (int i = 0; i < num_signals * length; ++i) {
        abs[i] = sqrt(hilbert_complex[i].x * hilbert_complex[i].x + hilbert_complex[i].y * hilbert_complex[i].y);
    }
}

void calculateDivideMean(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs, cufftComplex *divide_mean) {
    for (int i = 0; i < num_signals * length; ++i) {
        hilbert_complex[i].x /= abs[i];
        hilbert_complex[i].y /= abs[i];
    }
    calculateMean(hilbert_complex, length, num_signals, divide_mean);
    
}

void calculateMultiply(float *num1, float *num2, float *res, size_t length) {
    for (int i = 0; i < length; ++i) {
        res[i] = num1[i] * num2[i];
    }
}

void normalize(float *num, size_t length) {
    float maxVal = 0;
    for (size_t i = 0; i < length; ++i) {
        if (num[i] > maxVal) {
            maxVal = num[i];
        }
    }

    if (maxVal == 0) return;

    // Step 2: Divide each element by the maximum value to normalize
    for (size_t i = 0; i < length; ++i) {
        num[i] /= maxVal;
    }
}

void pws_stack(cufftComplex *hilbert_complex, const int length, const int num_signals, float *mean_amp, float *weighted_amp) {
    // mean_amp: 1024
    calculateMeanReal(hilbert_complex, length, num_signals, mean_amp);
    
    // abs_hilb_amp: 2 * 1024
    float *abs_hilb_amp = (float *)malloc(num_signals * length * sizeof(float));
    calculateABS(hilbert_complex, length, num_signals, abs_hilb_amp);

    // divede_mean: 1024
    cufftComplex *divide_mean = (cufftComplex *)malloc(length * sizeof(cufftComplex));
    calculateDivideMean(hilbert_complex, length, num_signals, abs_hilb_amp, divide_mean);

    // weight: 1024
    float *weight = (float *)malloc(length * sizeof(float));
    calculateABS(divide_mean, length, 1, weight);

    // weighted_amp: 1024
    calculateMultiply(mean_amp, weight, weighted_amp, length);

    // normalize mean_amp, weighted_amp
    normalize(mean_amp, length);
    normalize(weighted_amp, length);

    // free memory
    free(abs_hilb_amp);
    free(divide_mean);
    free(weight);

}

__global__ void Print_Complex(cufftComplex *dev_complex, const int LENGTH)
{
    printf("on GPU: \n");
    for (int i = 0; i < 500; i++) {
      printf("Real[%d]:%f, Img[%d]:%f\n", i, dev_complex[i].x, i, dev_complex[i].y); 
    }     
}

__global__ void initComplexArray(cuComplex *arr, int n) {
    for (int i = 0; i < n; i++) {
      arr[i].x = sin(i * 3.14 * 0.01);
      arr[i].y = 0.0;
    }
}