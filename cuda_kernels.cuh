#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < K)
    {
        float temp = 0.0f;
        for (int i = 0; i < N; i++)
        {
            temp += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = temp;
    }
}

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;
    for (int phase = 0; phase < (N + TILE_WIDTH - 1) / TILE_WIDTH; phase++)
    {
        if (row < M && (phase * TILE_WIDTH + threadIdx.x) < N)
        {
            tile_a[threadIdx.y][threadIdx.x] = A[row * N + (phase * TILE_WIDTH + threadIdx.x)];
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((phase * TILE_WIDTH + threadIdx.y) < N && col < K)
        {
            tile_b[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * K + col];
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            value += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < K)
    {
        C[row * K + col] = value;
    }
}
