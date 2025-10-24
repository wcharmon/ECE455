#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>

#define TILE_SIZE 16
#define MAT_DIM 1024

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
__global__ void mm_naive(const T* A, const T* B, T* C, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N;
    if (tid >= total_elements) return;

    int row = tid / N;
    int col = tid % N;

    T sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[tid] = sum;
}

template <typename T>
__global__ void mm_tiled(const T* A,const T* B, T* C, size_t N) {

    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < ((N + TILE_SIZE - 1) / TILE_SIZE); ++t) {

        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // wait for tiles to be filled

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        __syncthreads();

    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }

}

template <typename T>
void mm_host(const T* A, const T* B, T* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k)
                acc += static_cast<double>(A[i * N + k]) * static_cast<double>(B[k * N + j]);
            C[i * N + j] = static_cast<T>(acc);
        }
    }
}

template <typename T>
bool validate_results(const std::vector<T>& ref,
                      const std::vector<T>& gpu,
                      int N,
                      T rel_tol = static_cast<T>(1e-2)) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t idx = i * N + j;
            T diff = std::abs(ref[idx] - gpu[idx]);
            T denom = std::max(static_cast<T>(1.0), std::abs(ref[idx]));
            if (diff / denom > rel_tol) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << "CPU=" << ref[idx] << ", GPU=" << gpu[idx]
                          << ", rel_err=" << diff / denom << std::endl;
                return false;
            }
        }
    }
    return true;
}

template <typename KernelFunc, typename T>
float measure_kernel_time(KernelFunc kernel, int N, bool tiled) {
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(T) * N * N);
    cudaMalloc(&d_B, sizeof(T) * N * N);
    cudaMalloc(&d_C, sizeof(T) * N * N);

    dim3 threads, blocks;

    if (tiled) {
        threads = dim3(TILE_SIZE, TILE_SIZE);
        blocks = dim3((N + TILE_SIZE - 1) / TILE_SIZE,
                      (N + TILE_SIZE - 1) / TILE_SIZE);
    } else {
        int threadsPerBlock = 256;
        int totalThreads = N * N;
        blocks = dim3((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
        threads = dim3(threadsPerBlock);
    }

    // Warm-up
    if (tiled)
        mm_tiled<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else
        mm_naive<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(startEvent, 0);
    if (tiled)
        mm_tiled<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else
        mm_naive<T><<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float time_ms;
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time_ms;
}

int main() {
    using T = float;
    const int N = MAT_DIM;
    std::vector<T> h_A = create_rand_vector<T>(N * N);
    std::vector<T> h_B = create_rand_vector<T>(N * N);
    std::vector<T> h_C_ref(N * N, 0);
    std::vector<T> h_C_gpu_tiled(N * N, 0);
    std::vector<T> h_C_gpu_naive(N * N, 0);

    // reference implementation for validation
    std::cout << "Running CPU reference..." << std::endl;
    mm_host(h_A.data(), h_B.data(), h_C_ref.data(), N);

    T *data_A, *data_B, *data_C;
    cudaMalloc(&data_A, (sizeof(T) * N * N));
    cudaMalloc(&data_B, (sizeof(T) * N * N));
    cudaMalloc(&data_C, (sizeof(T) * N * N));

    cudaMemcpy(data_A, h_A.data(), sizeof(T) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(data_B, h_B.data(), sizeof(T) * N * N, cudaMemcpyHostToDevice);

    // run kernel naive
    int threadsPerBlock = 256;
    int totalThreads = N * N;
    dim3 blocks((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
    dim3 threads(threadsPerBlock);
    mm_naive<<<blocks, threads>>>(data_A, data_B, data_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu_naive.data(), data_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    // run kernel tiled
    dim3 threadsPer(TILE_SIZE, TILE_SIZE);
    dim3 blocksPer((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    mm_tiled<<<blocksPer, threadsPer>>>(data_A, data_B, data_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu_tiled.data(), data_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);


    // validate
    std::cout << "Validating tiled version..." << std::endl;
    bool ok_tiled = validate_results(h_C_ref, h_C_gpu_tiled, N);
    bool ok_naive = validate_results(h_C_ref, h_C_gpu_naive, N);

    if (!ok_tiled || !ok_naive) {
        std::cerr << "Validation FAILED." << std::endl;
        return 1;
    }
    std::cout << "Validation PASSED for both." << std::endl;
    
    // timing stuff
    float time_naive = measure_kernel_time<decltype(mm_naive<float>), float>(mm_naive<float>, N, false);
    float time_tiled = measure_kernel_time<decltype(mm_tiled<float>), float>(mm_tiled<float>, N, true);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Naive (global memory): " << time_naive << " ms" << std::endl;
    std::cout << "Tiled (shared memory): " << time_tiled << " ms" << std::endl;
    std::cout << "Speedup: " << time_naive / time_tiled << "x" << std::endl;

    cudaFree(data_A);
    cudaFree(data_B);
    cudaFree(data_C);


    return 0;


}