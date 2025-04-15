# CUDA GEMM implementation worklog
This is a follow-through implementation of the blog ["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm.

All experiments and profiling are performed on [Tensara](https://tensara.org/problems/matrix-multiplication).

# Kernel 1. Naive implementation
* [gemm/naive_gemm.cu](gemm/naive_gemm.cu)

A very naive implementation of the gemm kernel, each thread calculates their corresponding entry in the result matrix.
| GPU    	| T4      	| H100 	|
|--------	|---------	|------	|
| GFLOPS 	| Timeout 	|      	|

# Kernel 2. Naive implementation with global memory coalescing

* [gemm/naive_gemm_coalesce.cu](gemm/naive_gemm_coalesce.cu)

Naive GEMM, but global memory coalescing is utilized, so that the threads working on the same row can be packed into one warp, thus within warp broadcast accelerates the MIO.

```diff
- int row = blockDim.x*blockIdx.x+threadIdx.x;
- int col = blockDim.y*blockIdx.y+threadIdx.y;
+ int row = 32 * blockIdx.x + threadIdx.x / 32;
+ int col = 32 * blockIdx.y + threadIdx.x % 32;
```
| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 611.16 	| 5830.11 	|

# Kernel 3. Naive Shared Memory

* [gemm/shared_mem.cu](gemm/share_mem.cu)

Utilizing the shared memory naively. For some reason this version is even slower than Kernel 2. I guess the main reason is that there is still some issues concerning that the memory read can't be coalesced.

| GPU    	| T4      	| H100 	|
|--------	|---------	|------	|
| GFLOPS 	| Timeout 	|      	|

* [gemm/shared_mem_coalesce_2d_block.cu](gemm/shared_mem_coalesce_2d_block.cu)

| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 667.51 	| 6288.04 	|