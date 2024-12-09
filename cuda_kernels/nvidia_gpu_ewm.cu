#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

// CUDA kernel for element-wise multiplication
__global__ void elementwise_multiply_kernel(const double *a, const double *b, double *c, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        c[idx] = a[idx] * b[idx];
    }
}

// Host wrapper function for CUDA kernel
extern "C" void elementwise_multiply_gpu(const double *a, const double *b, double *c, size_t total_elements) {
    double *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t size = total_elements * sizeof(double);

    // Allocate device memory
    if (cudaMalloc((void **)&d_a, size) != cudaSuccess) {
        printf("CUDA malloc failed for d_a\n");
        return;
    }
    if (cudaMalloc((void **)&d_b, size) != cudaSuccess) {
        printf("CUDA malloc failed for d_b\n");
        cudaFree(d_a);
        return;
    }
    if (cudaMalloc((void **)&d_c, size) != cudaSuccess) {
        printf("CUDA malloc failed for d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }

    // Copy data from host to device
    if (cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA memcpy failed for d_a\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }
    if (cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA memcpy failed for d_b\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    // Configure CUDA kernel launch parameters
    size_t threads_per_block = 256;
    size_t blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    elementwise_multiply_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, total_elements);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    // Wait for device to finish and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    // Copy result back to host
    if (cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("CUDA memcpy failed for d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
