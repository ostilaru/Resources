#include<iostream>
#include<cuda_runtime.h>

const int matrixSize = 1024 * 36;

__global__ void matrixMultiplication(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrixSize && col < matrixSize) {
        int sum = 0;
        for (int k = 0; k < matrixSize; ++k) {
            sum += a[row * matrixSize + k] * b[k * matrixSize + col];
        }
        c[row * matrixSize + col] = sum;
    }
}

int main() {
    int *host_a = new int[matrixSize * matrixSize];
    int *host_b = new int[matrixSize * matrixSize];
    int *host_c = new int[matrixSize * matrixSize];

    // 初始化矩阵
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
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
    cudaMalloc((void**)&dev_a, matrixSize * matrixSize * sizeof(int));
    cudaMalloc((void**)&dev_b, matrixSize * matrixSize * sizeof(int));
    cudaMalloc((void**)&dev_c, matrixSize * matrixSize * sizeof(int));

    // 将数据从device复制到GPU
    cudaMemcpy(dev_a, host_a, matrixSize * matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, matrixSize * matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格
    dim3 gridDim(matrixSize / 32, matrixSize / 32);
    dim3 blockDim(32, 32);

    // 记录开始时间（仅GPU计算）
    cudaEventRecord(startGPU);

    // 调用GPU上的核函数
    matrixMultiplication<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // 在主机上等待GPU计算完成
    cudaDeviceSynchronize();

    // 记录结束时间（仅GPU计算）
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    // 将结果从GPU复制回device
    cudaMemcpy(host_c, dev_c, matrixSize * matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

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
