#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

// Kernel definition

// Kernel definition with optimization
__global__ void nvidia_gpu_dot(double *a, double *b, double *c, size_t rows_a, size_t cols_a, size_t cols_b) {
    __shared__ double tile_a[TILE_SIZE][TILE_SIZE + 1]; // Padding to avoid bank conflicts
    __shared__ double tile_b[TILE_SIZE][TILE_SIZE + 1];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (size_t tile_idx = 0; tile_idx < (cols_a + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
        if (row < rows_a && tile_idx * TILE_SIZE + threadIdx.x < cols_a)
            tile_a[threadIdx.y][threadIdx.x] = a[row * cols_a + tile_idx * TILE_SIZE + threadIdx.x];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0;

        if (col < cols_b && tile_idx * TILE_SIZE + threadIdx.y < cols_a)
            tile_b[threadIdx.y][threadIdx.x] = b[(tile_idx * TILE_SIZE + threadIdx.y) * cols_b + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Loop unrolling
        #pragma unroll
        for (size_t k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows_a && col < cols_b) {
        c[row * cols_b + col] = sum;
    }
}



extern "C" void dot_parallel_gpu_nvidia(double *a, double *b, double *c, size_t rows_a, size_t cols_a, size_t cols_b) {
    double *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t size_a = rows_a * cols_a * sizeof(double);
    size_t size_b = cols_a * cols_b * sizeof(double);
    size_t size_c = rows_a * cols_b * sizeof(double);

    if (cudaMalloc((void **)&d_a, size_a) != cudaSuccess) {
        printf("CUDA malloc failed for d_a\n");
        return;
    }
    if (cudaMalloc((void **)&d_b, size_b) != cudaSuccess) {
        printf("CUDA malloc failed for d_b\n");
        cudaFree(d_a);
        return;
    }
    if (cudaMalloc((void **)&d_c, size_c) != cudaSuccess) {
        printf("CUDA malloc failed for d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }

    if (cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA memcpy failed for d_a\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    if (cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA memcpy failed for d_b\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((cols_b + TILE_SIZE - 1) / TILE_SIZE, (rows_a + TILE_SIZE - 1) / TILE_SIZE);

    nvidia_gpu_dot<<<grid_dim, block_dim>>>(d_a, d_b, d_c, rows_a, cols_a, cols_b);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    if (cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("CUDA memcpy failed for d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


