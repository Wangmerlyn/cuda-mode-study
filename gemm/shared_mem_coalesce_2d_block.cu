#include <cuda_runtime.h>

__global__ void naive_mm(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k, int tile_w){
    int tile_row=threadIdx.y;
    int tile_col=threadIdx.x;
    int row = blockIdx.x*blockDim.x+tile_row;
    int col = blockIdx.y*blockDim.y+tile_col;
    extern __shared__ float As[];
    float* Bs=&As[tile_w*tile_w];
    
    float result=0.0f;
    for(int tile_idx=0;tile_idx<(k+tile_w-1)/tile_w;tile_idx++){
        As[tile_w*tile_row+tile_col]= tile_idx*tile_w+tile_col<k && row<m ? input_a[row*k+tile_idx*tile_w+tile_col]:0.0f;
        Bs[tile_w*tile_row+tile_col]= tile_idx*tile_w+tile_row<k && col<n ? input_b[(tile_idx*tile_w+tile_row)*n+col]:0.0f;
        __syncthreads();
        for(int i=0;i<tile_w;i++){
            result+=As[tile_row*tile_w+i]*Bs[i*tile_w+tile_col];
        }
        __syncthreads();
    }
    if(row<m&&col<n)output_c[row*n+col]=result;
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    int tile_w = 32;    
    // dim3 tpb = dim3(tile_w*tile_w);
    dim3 tpb=dim3(tile_w, tile_w);
    dim3 gridDim= dim3((m+tile_w-1)/tile_w,(n+tile_w-1)/tile_w);
    size_t shared = tile_w*tile_w*2*sizeof(float);
    naive_mm<<<gridDim, tpb, shared>>>(input_a, input_b, output_c, m, n, k, tile_w);
}