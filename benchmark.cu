#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

/ Include your implementation files
#include "cuda_kernels.cuh" // Contains CUDA kernel declarations/definitions
#include "openmp_kernels.h" // Contains OpenMP function declarations

    // Result structure
    struct BenchmarkResult
{
    std::string method;
    int size;
    int trial;
    double wall_ms;
    double kernel_ms;
    double memcpy_htod_ms;
    double memcpy_dtoh_ms;
    double gflops;
    double accuracy;
};

// Utility: Initialize matrix with random values
void initialize_matrix(float *mat, int size, unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size * size; i++)
    {
        mat[i] = dis(gen);
    }
}

// Utility: Compute relative error between two matrices
double compute_error(const float *C_ref, const float *C_test, int size)
{
    double max_error = 0.0;
    for (int i = 0; i < size * size; i++)
    {
        double error = std::abs(C_ref[i] - C_test[i]) / (std::abs(C_ref[i]) + 1e-7);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

// Utility: Compute GFLOPS
double compute_gflops(int N, double time_ms)
{
    double ops = 2.0 * N * N * N; // 2*N^3 operations
    return ops / (time_ms * 1e6); // Convert ms to seconds, then to GFLOPS
}

// CPU Baseline (naive) - used as reference
void matmul_cpu_reference(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Benchmark OpenMP implementations
BenchmarkResult benchmark_openmp(const std::string &method, const float *A, const float *B,
                                 float *C, int N, int trial, bool use_tiled = false, int tile_size = 64)
{
    BenchmarkResult result;
    result.method = method;
    result.size = N;
    result.trial = trial;
    result.memcpy_htod_ms = 0.0;
    result.memcpy_dtoh_ms = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    if (use_tiled)
    {
        matmul_openmp_tiled(A, B, C, N, tile_size);
    }
    else
    {
        matmul_openmp_naive(A, B, C, N);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    result.wall_ms = elapsed.count();
    result.kernel_ms = elapsed.count();
    result.gflops = compute_gflops(N, result.wall_ms);

    return result;
}

// Benchmark CUDA implementations
BenchmarkResult benchmark_cuda(const std::string &method, const float *h_A, const float *h_B,
                               float *h_C, int N, int trial, bool use_tiled = false)
{
    BenchmarkResult result;
    result.method = method;
    result.size = N;
    result.trial = trial;

    float *d_A, *d_B, *d_C;
    size_t bytes = N * N * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Measure H2D transfer
    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);

    cudaEventRecord(h2d_start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);

    float h2d_ms;
    cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);
    result.memcpy_htod_ms = h2d_ms;

    // Measure kernel execution
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaEventRecord(kernel_start);
    if (use_tiled)
    {
        matmul_cuda_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    }
    else
    {
        matmul_cuda_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    }
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
    result.kernel_ms = kernel_ms;

    // Measure D2H transfer
    cudaEvent_t d2h_start, d2h_stop;
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    cudaEventRecord(d2h_start);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);

    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);
    result.memcpy_dtoh_ms = d2h_ms;

    result.wall_ms = h2d_ms + kernel_ms + d2h_ms;
    result.gflops = compute_gflops(N, result.kernel_ms);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    return result;
}

// Main benchmark runner
void run_benchmarks(const std::vector<int> &sizes, int n_trials = 10)
{
    std::ofstream csv_file("benchmark_results.csv");
    csv_file << "method,size,trial,wall_ms,kernel_ms,memcpy_htod_ms,memcpy_dtoh_ms,gflops,accuracy\n";

    std::cout << std::fixed << std::setprecision(3);

    for (int N : sizes)
    {
        std::cout << "\n=== Benchmarking size N = " << N << " ===\n";

        // Allocate matrices
        size_t bytes = N * N * sizeof(float);
        float *A = (float *)aligned_alloc(64, bytes);
        float *B = (float *)aligned_alloc(64, bytes);
        float *C_ref = (float *)aligned_alloc(64, bytes);
        float *C_test = (float *)aligned_alloc(64, bytes);

        // Initialize
        initialize_matrix(A, N, 42);
        initialize_matrix(B, N, 123);

        // Compute reference (small sizes only for validation)
        if (N <= 512)
        {
            std::cout << "Computing reference solution...\n";
            matmul_cpu_reference(A, B, C_ref, N);
        }

        // Test each method
        std::vector<std::string> methods = {"OpenMP_Naive", "OpenMP_Tiled", "CUDA_Naive", "CUDA_Tiled"};

        for (const auto &method : methods)
        {
            std::cout << "Testing " << method << "...\n";

            // Warmup
            if (method == "OpenMP_Naive")
            {
                matmul_openmp_naive(A, B, C_test, N);
            }
            else if (method == "OpenMP_Tiled")
            {
                matmul_openmp_tiled(A, B, C_test, N, 64);
            }
            else if (method == "CUDA_Naive")
            {
                benchmark_cuda(method, A, B, C_test, N, -1, false);
            }
            else if (method == "CUDA_Tiled")
            {
                benchmark_cuda(method, A, B, C_test, N, -1, true);
            }

            // Validate accuracy (once)
            double error = 0.0;
            if (N <= 512)
            {
                if (method.find("OpenMP") != std::string::npos)
                {
                    error = compute_error(C_ref, C_test, N);
                }
                else
                {
                    // For CUDA, run one more time to get result in C_test
                    if (method == "CUDA_Naive")
                    {
                        benchmark_cuda(method, A, B, C_test, N, 0, false);
                    }
                    else
                    {
                        benchmark_cuda(method, A, B, C_test, N, 0, true);
                    }
                    error = compute_error(C_ref, C_test, N);
                }
                std::cout << "  Relative error: " << error << "\n";
            }

            // Run trials
            for (int trial = 0; trial < n_trials; trial++)
            {
                BenchmarkResult result;

                if (method == "OpenMP_Naive")
                {
                    result = benchmark_openmp(method, A, B, C_test, N, trial, false);
                }
                else if (method == "OpenMP_Tiled")
                {
                    result = benchmark_openmp(method, A, B, C_test, N, trial, true, 64);
                }
                else if (method == "CUDA_Naive")
                {
                    result = benchmark_cuda(method, A, B, C_test, N, trial, false);
                }
                else if (method == "CUDA_Tiled")
                {
                    result = benchmark_cuda(method, A, B, C_test, N, trial, true);
                }

                result.accuracy = error;

                // Write to CSV
                csv_file << result.method << ","
                         << result.size << ","
                         << result.trial << ","
                         << result.wall_ms << ","
                         << result.kernel_ms << ","
                         << result.memcpy_htod_ms << ","
                         << result.memcpy_dtoh_ms << ","
                         << result.gflops << ","
                         << result.accuracy << "\n";

                if (trial == 0)
                {
                    std::cout << "  Trial 0: " << result.kernel_ms << " ms, "
                              << result.gflops << " GFLOPS\n";
                }
            }
        }

        // Cleanup
        free(A);
        free(B);
        free(C_ref);
        free(C_test);
    }

    csv_file.close();
    std::cout << "\nResults saved to benchmark_results.csv\n";
}

int main(int argc, char **argv)
{
    // Print system info
    std::cout << "=== System Information ===\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "GPU Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";

    int driver, runtime;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::cout << "CUDA Driver: " << driver / 1000 << "." << (driver % 100) / 10 << "\n";
    std::cout << "CUDA Runtime: " << runtime / 1000 << "." << (runtime % 100) / 10 << "\n\n";

    // Matrix sizes to test
    std::vector<int> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    // If GPU has limited memory, remove larger sizes
    if (prop.totalGlobalMem < 4ULL * 1024 * 1024 * 1024)
    {
        sizes = {128, 256, 512, 1024};
    }

    int n_trials = 10;
    if (argc > 1)
    {
        n_trials = std::atoi(argv[1]);
    }

    std::cout << "Running " << n_trials << " trials for each configuration\n";

    run_benchmarks(sizes, n_trials);

    return 0;
}