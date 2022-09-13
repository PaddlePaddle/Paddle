![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Efficient GEMM in CUDA")

[README](/README.md#documentation) > **Efficient GEMM in CUDA**

# Efficient GEMM in CUDA

CUTLASS implements the hierarchically blocked structure described in 
[CUTLASS: Fast Linear Algebra in CUDA C++](https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/)
and the [CUTLASS GTC2018 talk](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf).

## Hierarchical Structure

The basic triple loop nest computing matrix multiply may be blocked and tiled to match
concurrency in hardware, memory locality, and parallel programming models. In CUTLASS, 
GEMM is mapped to NVIDIA GPUs with the structure illustrated by the following loop nest.

```c++
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling 
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            // 
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

This tiled loop nest targets concurrency among
- threadblocks
- warps
- CUDA and Tensor Cores

and takes advantage of memory locality within
- shared memory
- registers

The flow of data within this structure is illustrated below. 
This is the hierarchical GEMM computation embodied by CUTLASS. Each stage depicts a 
nested level of tiling which corresponds to a layer of concurrency within the CUDA execution model and to a 
level within the memory hierarchy, becoming increasingly finer moving left to right.

![ALT](/media/images/gemm-hierarchy-with-epilogue.png "Hierarchical GEMM in CUDA")


### Threadblock-level GEMM

Each threadblock computes its portion of the output GEMM by iteratively loading tiles of input
matrices and computing an accumulated matrix product. At the threadblock level, data is loaded from
global memory. The blocking strategy in general is key to achieving efficiency. However, there are
multiple conflicting goals that a programmer aims to achieve to strike a reasonable compromise. A
larger threadblock means fewer fetches from global memory, thereby ensuring that DRAM bandwidth
does not become a bottleneck. 

However, large threadblock tiles may not match the dimensions of the problem well. If either the
GEMM _M_ or _N_ dimension is small, some threads within the threadblock may not perform meaningful
work, as the threadblock may be partially outside the bounds of the problem. If both _M_ and _N_
are small while _K_ is large, this scheme may launch relatively few threadblocks and fail to
fully utilize all multiprocessors within the GPU. Strategies to optimize performance for this case
are described in the section [Parallelized Reductions](efficient_gemm.md#parallelized-reductions) 
which partition the GEMM K dimension across multiple threadblocks or multiple warps. These compute
matrix products in parallel which is then reduced to compute the result.

In CUTLASS, the dimensions of the threadblock tile are specified as `ThreadblockShape::{kM, kN, kK}`
and may be tuned to specialize the GEMM computation for the target processor and dimensions of
the GEMM problem.


### Warp-level GEMM

The warp-level GEMM maps to the warp-level parallelism within the CUDA execution model. Multiple
warps within a threadblock fetch data from shared memory into registers and perform computations.
Warp-level GEMMs may be implemented either by TensorCores issuing 
[mma.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) 
or [wmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma) 
instructions or by thread-level matrix computations issued to CUDA cores.
For maximum performance, access to shared memory should be bank conflict free. To maximize data
reuse within the warp, a large warp-level GEMM tile should be chosen.


### Thread-level GEMM

At the lowest level of blocking, each thread is responsible for processing a certain number of
elements. Threads cannot access each other's registers so we choose an organization that enables
values held in registers to be reused for multiple math instructions. This results in a 2D tiled
structure within a thread, in which each thread issues a sequence of independent math instructions
to the CUDA cores and computes an accumulated outer product.

SGEMM, IGEMM, HGEMM, and DGEMM are computed by SIMT math instructions issued by thread-level matrix multiply
procedures.


## Epilogue

The above code focuses only on the matrix multiply computation **C = AB** whose result is
held in the registers of each thread within the threadblock. The mapping of logical elements
in the output tile to each thread is chosen to maximize performance of the matrix multiply
computation but does not result in efficient, coalesced loads and stores to global memory.

The epilogue is a separate phase in which threads exchange data through shared memory then
cooperatively access global memory using efficient striped access patterns. It is also
the phase in which linear scaling and other elementwise operations may be conveniently
computed using the matrix product results as inputs.

CUTLASS defines several typical epilogue operations such as linear scaling and clamping,
but other device-side function call operators may be used to perform custom operations.

## Optimizations

The hierarchical structure described above yields an efficient mapping to the CUDA execution model and 
CUDA/TensorCores in NVIDIA GPUs. The following sections describe strategies for obtaining peak performance
for all corners of the design space, maximizing parallelism and exploiting data locality wherever possible.

### Pipelining

The blocked structure demands a large storage allocation within the registers of each CUDA thread. The
accumulator elements typically occupy at least half a thread's total register budget. Consequently, 
occupancy -- the number of concurrent threads, warps, and threadblocks -- is relatively low compared
to other classes of GPU workloads. This limits the GPUs ability to hide memory latency and other stalls
by context switching to other concurrent threads within an SM.

To mitigate the effects of memory latency, *software pipelining* is used to overlap memory accesses
with other computation within a thread. In CUTLASS, this is achieved by double buffering at the
following scopes

- **threadblock-scoped shared memory tiles:** two tiles are allocated within shared memory; one is used
  load data for the current matrix operation, while the other tile is used to buffer data loaded from
  global memory for the next mainloop iteration

- **warp-scoped matrix fragments:** two fragments are allocated within registers; one fragment is passed
  to CUDA and TensorCores during the current matrix computation, while the other is used to receive
  shared memory fetch returns for the next warp-level matrix operation

The efficient, pipelined mainloop body used in CUTLASS GEMMs is illustrated as follows.

![ALT](/media/images/software-pipeline.png "Software pipeline in CUTLASS")

### Threadblock Rasterization

To maximize reuse of data held in the last level cache, CUTLASS defines several functions to
affect the mapping of threadblocks to logical partitions of the GEMM problem. These map
consecutively launched threadblocks to packed two-dimensional regions of the partitioned GEMM
problem to increase the probability that these will access the same tiles of global memory at
approximately the same time.

Several functions are defined in [cutlass/gemm/threadblock_swizzle.h](/include/cutlass/gemm/threadblock/threadblock_swizzle.h).


### Parallelized Reductions

**Split K - reduction across threadblocks**

Matrix product computations expose parallelism among _O(MN)_ independent inner product
computations. For sufficiently large problem sizes, a GEMM kernel in CUTLASS may approach
the theoretical maximum computational throughput. For small problems, however, there are
too few threadblocks to efficiently occupy the entire GPU.

As a recourse, parallelizing the reduction performed during the inner product computation
enables more threadblocks to execute concurrently while still taking advantage of the throughput
benefits of large threadblock-level GEMM tiles.

CUTLASS implements parallel reductions across threadblocks by partitioning the GEMM _K_ dimension
and launching an additional set of threadblocks for each partition. Consequently, we refer to
this strategy within CUTLASS as "parallel reduction splitK." The "parallel reduction splitK" in cutlass 
requires the execution of 2 kernels. The first one is called partitionedK GEMM. The second one is called 
batched reduction.

The partitionedK GEMM is very similar to one flavor of batched strided GEMM. Instead of requiring users 
to specify the problem size of each batch, partitionedK GEMM asks for the overall problem size and the 
number of partition that will be applied along K dimension for operand A and B. For example, parameters o
f m=128, n=128, k=4096 and partition=16 will result in 16 batched strided GEMMs with each batch of 
m=128, n=128, k=256. PartitionedK also allows scenario where k is not divisible by partition count. 

For example, parameters of m=128, n=128, k=4096 and partition=20 will result in 20 batched strided GEMMs 
with the first 19 batches of m=128, n=128, k=4096/20=204 and the last batch of m=128, n=128, k=220.

The batched reduction kernel will further perform reduction along the K-dimension. Thus, the input of 
the batched reduction kernel is the output (C) of partitionedK GEMM. An workspace memory is managed by 
the users to store this intermediate results.

**Sliced K - reduction across warps**

Similar to the split-k scenario, sliced-k aims at improving the efficiency of kernels with smaller M, N,
 but large K dimensions. In general at the thread-block level, the parameters CtaTileN, CtaTileM expose parallelism 
by partitioning the the work the among warps, and larger warpTiles expose better ILP (Instruction 
level parallelism) and reuse, but it also limits the number of warps running per thread-block, which reduces efficiency.

So in order to improve efficiency in such scenarios, partitioning the warpTiles also along ctaTileK helps improve the utilization 
of the underlying hardware by allowing more warps to run concurrently in a CTA.  Now, since sliced-k kernels breaks 
down a thread-blocks's computation among participating warps not just among the CtaTileN, CtaTileM dimension, 
but also the CtaTileK dimension it entails a small cost in form of a reduction which has to happen at the end among the 
participating warps - since each warp now owns a partial sum (since they compute using only a "slice" of ctaTileK). 

# Resources

The following additional resources describe design and implementation details of GEMMs
targeting NVIDIA GPUs.

- [Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100.](https://www.nvidia.com/en-us/gtc) (SR 21745)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/)
- [CUTLASS: SOFTWARE PRIMITIVES FOR DENSE LINEAR ALGEBRA AT ALL LEVELS AND SCALES WITHIN CUDA](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s8854-cutlass%3a+software+primitives+for+dense+linear+algebra+at+all+levels+and+scales+within+cuda)
- [Programming Tensor Cores: NATIVE VOLTA TENSOR CORES WITH CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
- [CUDA Programming Guide: warp matrix functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Matrix Multiply Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

# Copyright

Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
