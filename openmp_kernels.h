#include <omp.h>

// Naive OpenMP matrix multiplication
// A, B, C are row-major: C = A * B
// A: N x N, B: N x N, C: N x N
void matmul_openmp_naive(const float *A, const float *B, float *C, int N)
{
#pragma omp parallel for
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

// Tiled/Blocked OpenMP matrix multiplication
// Uses cache blocking to improve locality
// TILE_SIZE should be tuned based on cache size (typically 32, 64, or 128)
void matmul_openmp_tiled(const float *A, const float *B, float *C, int N, int TILE_SIZE = 64)
{
// Initialize C to zero
#pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
        C[i] = 0.0f;
    }

// Tiled multiplication
#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += TILE_SIZE)
    {
        for (int jj = 0; jj < N; jj += TILE_SIZE)
        {
            for (int kk = 0; kk < N; kk += TILE_SIZE)
            {
                // Compute bounds for this tile
                int i_end = (ii + TILE_SIZE < N) ? ii + TILE_SIZE : N;
                int j_end = (jj + TILE_SIZE < N) ? jj + TILE_SIZE : N;
                int k_end = (kk + TILE_SIZE < N) ? kk + TILE_SIZE : N;

                // Multiply tiles
                for (int i = ii; i < i_end; i++)
                {
                    for (int j = jj; j < j_end; j++)
                    {
                        float sum = 0.0f;
                        for (int k = kk; k < k_end; k++)
                        {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

// Alternative tiled version with better memory access pattern
// (inner loops iterate over contiguous memory)
void matmul_openmp_tiled_optimized(const float *A, const float *B, float *C, int N, int TILE_SIZE = 64)
{
// Initialize C to zero
#pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
        C[i] = 0.0f;
    }

#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += TILE_SIZE)
    {
        for (int jj = 0; jj < N; jj += TILE_SIZE)
        {
            for (int kk = 0; kk < N; kk += TILE_SIZE)
            {
                int i_end = (ii + TILE_SIZE < N) ? ii + TILE_SIZE : N;
                int j_end = (jj + TILE_SIZE < N) ? jj + TILE_SIZE : N;
                int k_end = (kk + TILE_SIZE < N) ? kk + TILE_SIZE : N;

                for (int i = ii; i < i_end; i++)
                {
                    for (int k = kk; k < k_end; k++)
                    {
                        float a_val = A[i * N + k];
#pragma omp simd
                        for (int j = jj; j < j_end; j++)
                        {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}