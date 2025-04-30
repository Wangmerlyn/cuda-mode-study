#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Function to check CUDA errors
inline void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to initialize matrix with random values
void initializeMatrix(float *matrix, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            float rand_number = static_cast<float>(rand() % 10);
            matrix[i * n + j] = rand_number; // Random values between 0 and 9
            // matrix[i * n + j] = static_cast<float>((i * n + j) / 10.0f); // Sequential values for easier debugging
        }
    }
}

// Matrix multiplication kernel (your implementation)
__global__ void mm_1dtiling(const float *input_a, const float *input_b, float *output_c, size_t m, size_t n, size_t k)
{
    const unsigned int BM = 64, BN = 64, BK = 8;
    const uint TM = 8; // TM stands for thread mapping, each thread will compute TM elements
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
    assert(totalResults == blockDim.x * TM);
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

// cuBLAS Matrix multiplication
void cublasMatMul(cublasHandle_t handle, float *d_A, float *d_B, float *d_C, size_t m, size_t n, size_t k)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS SGEMM (single-precision general matrix multiply)
    // cublasStatus_t status = cublasSgemm(handle,
    //                                     CUBLAS_OP_T, CUBLAS_OP_T,
    //                                     m, n, k,
    //                                     &alpha,
    //                                     d_A, k,
    //                                     d_B, n,
    //                                     &beta,
    //                                     d_C, m);
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        d_B, n,
                                        d_A, k,
                                        &beta,
                                        d_C, n);
    // convert d_C to row major
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error during matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to compare matrices (for testing)
bool compareMatrices(const float *A, const float *B, size_t m, size_t n, float tolerance = 1e-5f)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            if (fabs(A[i * n + j] - B[i * n + j]) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

// Test case
int main()
{
    size_t m = 232;  // Rows of A and C
    size_t n = 3321; // Columns of B and C
    size_t k = 4391; // Columns of A and rows of B

    // Allocate memory for matrices on host
    float *h_A = new float[m * k];
    float *h_B = new float[k * n];
    float *h_C = new float[m * n];           // This will hold the result of kernel
    float *h_C_reference = new float[m * n]; // This will hold the result of cuBLAS

    // Initialize matrices with random values
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);
    // log A, B

    // std::cout << "Matrix A:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < k; ++j)
    //     {
    //         std::cout << " " << h_A[i * k + j];
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Matrix B:" << std::endl;
    // for (size_t i = 0; i < k; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         std::cout << h_B[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Matrix C:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         std::cout << h_C[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Allocate memory for matrices on device
    float *d_A;
    float *d_B;
    float *d_C;
    checkCudaErrors(cudaMalloc(&d_A, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Set up cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    checkCudaErrors(cudaMemset(d_C, 0.0f, m * n * sizeof(float)));

    // Perform matrix multiplication using cuBLAS
    cublasMatMul(handle, d_A, d_B, d_C, m, n, k);

    // move d_A back to host
    checkCudaErrors(cudaMemcpy(h_A, d_A, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    // log d_A
    // std::cout << "Matrix A after kernel:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < k; ++j)
    //     {
    //         std::cout << h_A[i * k + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Copy the result from device to host
    checkCudaErrors(cudaMemcpy(h_C_reference, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // print h_C_reference
    // std::cout << "cuBLAS result:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         std::cout << h_C_reference[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Perform matrix multiplication using your kernel
    dim3 blockDim = dim3(64 * 8, 1, 1);
    dim3 gridDim = dim3((n + 63) / 64, (m + 63) / 64, 1);
    // reinitialize d_C
    checkCudaErrors(cudaMemset(d_C, 0, m * n * sizeof(float)));
    mm_1dtiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

    // Copy the result from device to host (your kernel result)
    checkCudaErrors(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // output custom kernel result
    // std::cout << "Custom kernel result:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         std::cout << h_C[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // // output cuBLAS result
    // std::cout << "cuBLAS result:" << std::endl;
    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         std::cout << h_C_reference[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // check the errors when kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Compare results
    if (compareMatrices(h_C, h_C_reference, m, n))
    {
        std::cout << "Test passed: Both results are the same!" << std::endl;
    }
    else
    {
        std::cout << "Test failed: The results differ!" << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_reference;
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    cublasDestroy(handle);

    return 0;
}