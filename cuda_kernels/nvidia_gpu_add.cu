#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matrix_add_kernel(double *a, double *b, double *c, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void gpu_matrix_add(double *a, double *b, double *c, size_t rows, size_t cols) {
    double *d_a, *d_b, *d_c;
    size_t size = rows * cols * sizeof(double);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, rows, cols);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
