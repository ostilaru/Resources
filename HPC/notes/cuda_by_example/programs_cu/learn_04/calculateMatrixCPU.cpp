#include<iostream>
#include<chrono>

const int matrixSize = 1024 * 4;

void matrixMultiplicationCPU(int *a, int *b, int *c) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            int sum = 0;
            for (int k = 0; k < matrixSize; ++k) {
                sum += a[i * matrixSize + k] * b[k * matrixSize + j];
            }
            c[i * matrixSize + j] = sum;
        }
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

    // 创建计时器
    auto start = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------
    // 调用CPU上的核函数
    matrixMultiplicationCPU(host_a, host_b, host_c);

    // ---------------------------------------------------------------
    // 计时结束
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // 打印计算时间
    std::cout << "CPU calculation time: " << duration.count() << " milliseconds" << std::endl;

    // 释放分配的内存
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}
