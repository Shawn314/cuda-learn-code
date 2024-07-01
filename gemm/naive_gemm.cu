#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void sgemm_naive(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Host function to initialize matrices and call the kernel
void test_sgemm_naive() {
    // Matrix dimensions
    int M = 10240; // Number of rows in A and C
    int N = 20480; // Number of columns in B and C
    int K = 50120; // Number of columns in A and rows in B

    // Scalars
    float alpha = 1.0f;
    float beta = 1.0f;

    // Allocate host memory
    std::vector<float> h_A(M * K, 1.0f); // Initialize A with 1.0
    std::vector<float> h_B(K * N, 1.0f); // Initialize B with 1.0
    std::vector<float> h_C(M * N, 0.0f); // Initialize C with 0.0

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, beta, d_A, d_B, d_C);    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Time to generate:  %3.5f ms \n", time);
 
    // Copy result from device to host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void warm_up() {
    printf("warming up gpu...");
    for(int i = 0; i < 2; i++) {
        test_sgemm_naive();
    }
    printf("warm up gpu done!");
}

int main() {
    warm_up();
    test_sgemm_naive();
    return 0;
}


