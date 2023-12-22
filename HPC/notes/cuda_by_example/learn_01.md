
# 1. 早期的 GPU 计算

早期的 GPU 计算使用起来非常复杂，受限于标准图形接口，例如 `OpenGL` 和 `DirectX`，是与 GPU 交互的唯一方式。


# 2. CUDA
2006年11月，NVIDIA 公布了业界第一个 `DirectX 10` GPU，即 `GeForce 8800GTX`。

在之前的图形处理架构中，计算资源划分为顶点着色器和像素着色器，而 `CUDA` 架构则不同，它包含了一个统一的着色器流水线，使得执行通用计算的程序能够对芯片上的每个 ALU 进行排列。此外，GPU 上的执行单元不仅能任意地读写内存，同时还能访问由软件管理的缓存，也称为共享内存。

# 3. CUDA 编程简介
```cpp
#include <iostream>

__global__ void kernel(void) {

}

int main() {
    kernel<<<1, 1>>>();
    printf("Hello, World! \n");
    return 0;
}
```

```cpp
#include <iostream>
#include "book.h"
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int)));

    add<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("2 + 7 = %d\n", c);

    return 0;
}

```

## 3.1 查询设备
```cpp
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error == cudaSuccess) {
        printf("Number of CUDA devices: %d\n", deviceCount);
    } else {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
    }

    return 0;
}
```
`cudaGetDeviceCount()` 函数是CUDA（Compute Unified Device Architecture）编程接口提供的一个函数，它用于获取系统中可用的GPU设备数量。

```cpp
int main() {
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for(int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

        // Print device name
        printf("%s\n", prop.name);
    }
}
```

## 3.2 GPU 上进行矢量求和
```cpp
#define N 10
int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // 在 GPU 上分配内存
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    // 在 cpu 上为数组 a, b 赋值
    for(int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    // 将数组 a, b 复制到 GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // 将数组 c 从 GPU 复制到 cpu
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // 显示结果
    for(int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // 释放在 GPU 上分配的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x; // 获取当前线程块的索引
    if(tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
```