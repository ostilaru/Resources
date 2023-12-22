#include<iostream>
#include<cuda_runtime.h>

__global__ void addWithCuda(int *a, int *b, int *c) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    int idx = threadId + blockId * (blockDim.x * blockDim.y);

    c[idx] = a[idx] + b[idx];
}

const int arraySize = 1024 * 1024 * 24;

int main() {
    int *host_a = new int[arraySize];
    int *host_b = new int[arraySize];
    int *host_c = new int[arraySize];

    // 初始化数组
    for(int i = 0; i < arraySize; i++) {
        host_a[i] = i;
        host_b[i] = i * 2;
    }

    // 创建 CUDA 事件对象用于计时
    cudaEvent_t startTotal, stopTotal, startGPU, stopGPU;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // 记录开始时间（总的开销）
    cudaEventRecord(startTotal);

    // ---------------------------------------------------------------
    // 在GPU上分配内存
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    // 将数据从device复制到GPU
    cudaMemcpy(dev_a, host_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格
    dim3 gridDim(24);
    dim3 blockDim(1024);
    dim3 threads(1024);

    // 记录开始时间（仅GPU计算）
    cudaEventRecord(startGPU);

    // 调用GPU上的核函数
    addWithCuda<<<blockDim, threads>>>(dev_a, dev_b, dev_c);

    // 在主机上等待GPU计算完成
    cudaDeviceSynchronize();

    // 记录结束时间（仅GPU计算）
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    // 将结果从GPU复制回device
    cudaMemcpy(host_c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------------
    // 记录结束时间（总的开销）
    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);

    // 计算执行时间（总的开销）
    float millisecondsTotal = 0;
    cudaEventElapsedTime(&millisecondsTotal, startTotal, stopTotal);

    // 计算执行时间（仅GPU计算）
    float millisecondsGPU = 0;
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

    std::cout << "Total time: " << millisecondsTotal << " milliseconds" << std::endl;
    std::cout << "GPU calculation time: " << millisecondsGPU << " milliseconds" << std::endl;

    // 释放分配的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}
