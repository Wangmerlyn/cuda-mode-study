#include <cuda_runtime.h>
constexpr int TILE_SIZE = 32;

__global__ void naive_mm(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k){
    __shared__ float M_TILE[TILE_SIZE][TILE_SIZE];
    __shared__ float N_TILE[TILE_SIZE][TILE_SIZE];

    int icol = threadIdx.x;
    int irow = threadIdx.y;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for(int k_idx = 0; k_idx < (k + TILE_SIZE - 1) / TILE_SIZE; k_idx++){
        M_TILE[irow][icol] = (k_idx * TILE_SIZE + icol < k && row < m) ? input_a[row * k + k_idx * TILE_SIZE + icol] : 0.0f;
        N_TILE[irow][icol] = (k_idx * TILE_SIZE + irow < k && col < n) ? input_b[(k_idx * TILE_SIZE + irow) * n + col] : 0.0f;
        __syncthreads();

        for(int k_idx = 0; k_idx < TILE_SIZE; k_idx++){
            sum += M_TILE[irow][k_idx] * N_TILE[k_idx][icol];
        }
        __syncthreads();
    }
    
    if(row < m && col < n){
        output_c[row * n + col] = sum;
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {  
    dim3 dimBlock = dim3(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid = dim3((n+TILE_SIZE-1)/TILE_SIZE, (m+TILE_SIZE-1)/TILE_SIZE);
    naive_mm<<<dimGrid,dimBlock>>>(input_a, input_b, output_c, m, n, k);
}