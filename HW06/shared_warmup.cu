#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>

#define BLOCK_DIM 256
#define N (1 << 16)

template <typename T>
std::vector<T> create_rand_vector(size_t n, T min_val = 0, T max_val = 50) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    std::vector<T> vec(n);
    for (size_t i = 0; i < n; ++i)
        vec[i] = static_cast<T>(dist(e));
    return vec;
}


template <typename T>
__global__ void square_kernel(const T* in, T* out, size_t size) {

    __shared__ T tile[BLOCK_DIM];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    tile[threadIdx.x] = in[idx];
    __syncthreads();

    tile[threadIdx.x] = tile[threadIdx.x] * tile[threadIdx.x];
    __syncthreads();

    out[idx] = tile[threadIdx.x];

}

int main() {
    using T = float;
    std::vector<T> h_in = create_rand_vector<T>(N);
    std::vector<T> h_out(N);

    T *data_in, *data_out;
    cudaMalloc(&data_in, sizeof(T) * N);
    cudaMalloc(&data_out, sizeof(T) * N);
    cudaMemcpy(data_in, h_in.data(), sizeof(T) * N, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid((N + BLOCK_DIM - 1) / BLOCK_DIM);
    square_kernel<<<blocksPerGrid, threadsPerBlock>>>(data_in, data_out, N);

    cudaMemcpy(h_out.data(), data_out, sizeof(T) * N, cudaMemcpyDeviceToHost);
    cudaFree(data_in);
    cudaFree(data_out);

    for (size_t i = 0; i < 5; ++i) {
        std::cout << "in[" << i << "] = " << h_in[i] << ", out[" << i << "] = " << h_out[i] << "\n";
    }

    return 0;


}