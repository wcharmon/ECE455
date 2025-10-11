#include <stdio.h>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    int N = 1000000;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int half = N/2;
    size_t half_size = size / 2;

    cudaMemcpyAsync(d_A, h_A, half_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, half_size, cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(d_A + half, h_A + half, half_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B + half, h_B + half, half_size, cudaMemcpyHostToDevice, stream2);

    int threads = 256;
    int blocks_half = (half + threads - 1) / threads;
    vector_add<<<blocks_half, threads, 0, stream1>>>(d_A, d_B, d_C, half);
    vector_add<<<blocks_half, threads, 0, stream1>>>(d_A + half, d_B + half, d_C + half, half);

    cudaMemcpyAsync(C, d_C, half_size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(C + half, d_C + half, half_size, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("C[0] = %f, C[N-1] = %f\n", h_C[0], C[N - 1]);

    cudaStreamDestory(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;

}