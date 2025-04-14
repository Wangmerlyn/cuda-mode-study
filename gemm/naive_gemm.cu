#include <cuda_runtime.h>

__global__ void naive_mm(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k){
    int row = blockDim.x*blockIdx.x+threadIdx.x;
    int col = blockDim.y*blockIdx.y+threadIdx.y;

    if(row<m && col < n){
        output_c[row*n+col] = 0.0f;
        for(int i=0;i<k;i++){
            output_c[row*n+col] += input_a[row*n+i]*input_b[i*k+col];
        }
    }
    
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {    
    dim3 tpb = dim3(32,32);
    dim3 gridDim= dim3((m+32-1)/32,(n+32-1)/32);
    naive_mm<<<gridDim, tpb>>>(input_a, input_b, output_c, m, n, k);
}
