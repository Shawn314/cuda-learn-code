#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
__global__ void exampleKernel() {
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int global_id = thread_id + block_id * (blockDim.x * blockDim.y);

    printf("Thread ID: %d, Block ID: %d, Global ID: %d\n", thread_id, block_id, global_id);
}

int main() {
    dim3 threadsPerBlock(16, 16);  // 每个线程块包含 16x16 个线程
    dim3 numBlocks(4, 4);          // 网格包含 4x4 个线程块

    exampleKernel<<<numBlocks, threadsPerBlock>>>();

    cudaDeviceSynchronize();
    return 0;
}
