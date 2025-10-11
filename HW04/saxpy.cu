#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_x, *h_y, *d_x, *d_y;

    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);
    for (int i = 0; i < N; i++){
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0F, d_x, d_y);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    printf("y[0] = %f\n", h_y[0]);

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}