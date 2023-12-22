#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void mykernel(int* cnt)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    printf("Hello from device! Block ID: %d, Thread ID: %d\n", blockId, threadId);

    // 在每个线程中递增计数器
    atomicAdd(cnt, 1);
}

int main()
{
    int cnt = 0;
    int* d_cnt; // 用于在GPU上分配内存的指针

    // 分配GPU上的内存
    cudaMalloc((void**)&d_cnt, sizeof(int));

    // 将主机上的cnt复制到GPU上的d_cnt
    cudaMemcpy(d_cnt, &cnt, sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格
    dim3 blockDim(16, 16);
    dim3 gridDim(16, 16);

    // 调用GPU上的核函数
    mykernel<<<gridDim, blockDim>>>(d_cnt);

    // 在主机上等待GPU计算完成
    cudaDeviceSynchronize();

    // 将结果从GPU上的d_cnt复制到主机上的cnt
    cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    cout << "Total number of threads: " << cnt << endl;

    // 释放GPU上的内存
    cudaFree(d_cnt);

    return 0;
}
