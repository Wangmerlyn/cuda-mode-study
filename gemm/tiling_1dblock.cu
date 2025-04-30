#include <cuda_runtime.h>
#include <cassert>
#define cdiv(a, b) ((a + b - 1) / b)

constexpr unsigned int BM = 64, BN = 64, BK = 8;
const uint TM = 8; // TM stands for thread mapping, each thread will compute TM elements

__global__ void mm_1dtiling(const float *input_a, const float *input_b, float *output_c, size_t m, size_t n, size_t k)
{
    // const unsigned int BM = 64, BN = 64, BK = 8;
    // const uint TM = 8; // TM stands for thread mapping, each thread will compute TM elements
    // load As, Bs
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    // here the access for row and col are reversed
    // in this way the adjacent block access the same row of A and access the column of B sequentially
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    const unsigned totalResults = BM * BN;
    // this block produces BM*BM results
    // each thread computes TM results
    // assert(totalResults == blockDim.x * TM);
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;
    float threadResult[TM] = {0.0f};
    for (uint bkIdx = 0; bkIdx < k; bkIdx += BK)
    {
        // TODO: add the code for boundary check
        // As[innerRowA * BK + innerColA] =  input_a[(cRow * BM + innerRowA) * k + bkIdx + innerColA];
        As[innerRowA * BK + innerColA] = (cRow * BM + innerRowA) < m && bkIdx + innerColA < k ? input_a[(cRow * BM + innerRowA) * k + bkIdx + innerColA] : 0.0f;
        // Bs[innerRowB * BN + innerColB] = input_b[(bkIdx + innerRowB) * n + cCol * BN + innerColB];
        Bs[innerRowB * BN + innerColB] = (bkIdx + innerRowB) < k && cCol * BN + innerColB < n ? input_b[(bkIdx + innerRowB) * n + cCol * BN + innerColB] : 0.0f;
        __syncthreads();
        for (uint j = 0; j < BK; ++j)
        {
            float tmpBs = Bs[j * BN + threadCol];
            for (uint i = 0; i < TM; ++i)
            {
                threadResult[i] += As[(threadRow * TM + i) * BK + j] * tmpBs;
            }
        }
        __syncthreads();
    }
    for (uint i = 0; i < TM; ++i)
    {
        // output_c[(cRow * BM + threadRow * TM + i) * n + cCol * BN + threadCol] = threadResult[i];
        float *C_shift = output_c + cRow * BM * n + cCol * BN;
        if (threadRow * TM + i + cRow * BM < m && cCol * BN + threadCol < n)
        {
            C_shift[(threadRow * TM + i) * n + threadCol] = threadResult[i];
        }
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float *input_a, const float *input_b, float *output_c, size_t m, size_t n, size_t k)
{
    // dim3 blockDim = dim3(64 * 8, 1, 1);
    dim3 blockDim = dim3(BM * BK, 1, 1);
    // dim3 gridDim = dim3(cdiv(n, 64), cdiv(m, 64), 1);
    dim3 gridDim = dim3(cdiv(n, BN), cdiv(m, BM), 1);
    // call kernel
    mm_1dtiling<<<gridDim, blockDim>>>(input_a, input_b, output_c, m, n, k);
}