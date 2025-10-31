#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <iomanip>
#include <chrono>

#define BLOCK_DIM 256
#define N (1 << 24) // a big number

// rand generator
std::vector<int> create_rand_vector(size_t n, int min_val = 0, int max_val = 100) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    std::vector<int> vec(n);
    for (size_t i = 0; i < n; ++i)
        vec[i] = dist(e);
    return vec;
}

// cpu reference
int reduce_ref(const std::vector<int>& data) {
    long long sum = 0;
    for (auto v : data)
        sum += v;
    return static_cast<int>(sum);
}


// kernel
__global__ void reduce_naive(const int* in, int* out, size_t size){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        atomicAdd(out, in[id]);
    }
}

int main(){

    std::vector<int> h_in = create_rand_vector(N);

    std::cout << "Testing GPU Kernel\n";
    // cpu version
    int ref = reduce_ref(h_in);

    // gpu version

    int *d_in, *d_out;
    int gpu_result = 0;

    cudaMalloc(&d_in, sizeof(int) * N);
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    dim3 threads(BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM);

    reduce_naive<<<blocks, threads>>>(d_in, d_out, N);

    cudaMemcpy(&gpu_result, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    if (gpu_result == ref) {
        std::cout << "Validation failed, GPU implementation does not match CPU implementation \n";
        std::cout << "GPU = " << gpu_result << ", CPU = " << ref << "\n"; 
    }
    else {
        std::cout << "Validation Passed!!!!!!!!!!!!!!!!!!!!!\n";
        std::cout << "GPU = " << gpu_result << ", CPU = " << ref << "\n"; 
    }
    

    return 0;

}