# CUDA GEMM implementation worklog
This is a follow-through implementation of the blog ["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm.

All experiments and profiling are performed on [Tensara](https://tensara.org/problems/matrix-multiplication).

# Kernel 1. Naive implementation
* [naive_gemm.cu](naive_gemm.cu)

A very naive implementation of the gemm kernel, each thread calculates their corresponding entry in the result matrix.
| GPU    	| T4      	| H100 	|
|--------	|---------	|------	|
| GFLOPS 	| Timeout 	|      	|

# Kernel 2. Naive implementation with global memory coalescing

* [naive_gemm_coalesce.cu](naive_gemm_coalesce.cu)

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

* [shared_mem.cu](share_mem.cu)

Utilizing the shared memory naively. For some reason this version is even slower than Kernel 2. I guess the main reason is that there is still some issues concerning that the memory read can't be coalesced.

| GPU    	| T4      	| H100 	|
|--------	|---------	|------	|
| GFLOPS 	| Timeout 	|      	|

* [shared_mem_coalesce_2d_block.cu](shared_mem_coalesce_2d_block.cu)

| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 667.51 	| 6288.04 	|

Later I was curious if combining this the above shared mem optimization approach can be push further with the same technique we used in Kernel 2. global memory coalescing, so I made this one.

* [shared_mem_coalesce.cu](shared_mem_coalesce.cu)

```diff
- int tile_row=threadIdx.y;
- int tile_col=threadIdx.x;
- int row = blockIdx.x*blockDim.x+tile_row;
- int col = blockIdx.y*blockDim.y+tile_col;
// tpb=tile_w*tile_w;
+ int tile_row = threadIdx.x / tile_w;
+ int tile_col = threadIdx.x % tile_w;
+ int row = blockIdx.x * tile_w + tile_row;
+ int col = blockIdx.y * tile_w + tile_col;
```

| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 671.01 	| 6350.95 	|

> **Note:** The `tile_w` is better not set as a function parameter, rather use it as a constexpr or template parameter is the way to go. This is because if `tile_w` is used as a function parameter, compiler can't determine the size of the share memory at compile time. see [cuda mode lecture 5](https://youtu.be/wVsR-YhaHlM?si=x8zn3UBIJZxJXycq&t=3171) for detailed explainations.

> after the `tile_w` fix

| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 881.68	| 8656.04 	|

# Kernel 4. Shared Memory with 1d Tiling

* [tiling_1dblock.cu](tiling_1dblock.cu)

The shared memory mapping for $A$ and $B$ are in the shape of $BM \times BK$ and $BK \times BN$.

In this implementation, $BN = BM$, and there are $BM \times BK$ threads in one block.

Each thread produces $TM$ results.

This gives us a total of:
$$
BM \times BK \times TM
$$
results in one block, and since $BM \times BK \times TM = BM \times BN$, the total matches the output tile size.

| GPU    	| T4     	| H100    	|
|--------	|--------	|---------	|
| GFLOPS 	| 1822.53	| 16038.13 	|