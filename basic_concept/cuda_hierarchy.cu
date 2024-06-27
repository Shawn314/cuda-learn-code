#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
__global__ void exampleKernel() {
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;         // 在线程块中的线程id
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;             // 在线程块中计算block的id
    int global_id = thread_id + block_id * (blockDim.x * blockDim.y); // 计算线程在所有线程中的全局id，其中（blockDim.x * blockDim.y）为一个线程块中拥有的线程数

    printf("Thread ID: %d, Block ID: %d, Global ID: %d\n", thread_id, block_id, global_id);
}

int main() {
    dim3 threadsPerBlock(16, 16);  // 每个线程块包含 16x16 个线程
    dim3 numBlocks(4, 4);          // 网格包含 4x4 个线程块
    /*
        gridDim.x = 4
        gridDim.y = 4
        blockDim.x = 16
        blockDim.y = 16
    */
    exampleKernel<<<numBlocks, threadsPerBlock>>>();

    cudaDeviceSynchronize();
    return 0;
}
