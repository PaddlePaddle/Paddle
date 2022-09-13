![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Profiler")

[README](/README.md#documentation) > **CUTLASS Profiler**

# CUTLASS Profiler

The CUTLASS Profiler is a command-line driven test and profiling environment for CUTLASS computations
defined in the CUTLASS Instance Library. The CUTLASS Profiler is capable of executing each GEMM, Sparse Gemm, 
Conv2d, and Conv3d kernel.

The CUTLASS Profiler may be compiled with:
```bash
$ make cutlass_profiler -j
```

To limit compilation time, only one tile size (typically 128x128) is instantiated for each data type, 
math instruction, and layout. To instantiate all sizes, set the following environment variable when running CMake from an 
empty `build/` directory.
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all  -DCUTLASS_UNITY_BUILD_ENABLED=ON
...
$ make cutlass_profiler -j
```
Enabling the unity build places multiple kernel instances in one compilation unit, thereby reducing size of the compiled
binary and avoiding linker limitations on some platforms.

The CUTLASS Profiler sources are stored in 
```bash
tools/
  profiler/
```

The CUTLASS Profiler usage statement may be obtained by executing `cutlass_profiler --help` and appears as follows.
```bash
CUTLASS Performance Tool
usage:

    cutlass_profiler [options]

  --help

  --mode=<string>                                  Cutlass profiler execution mode.
                                                    --mode=profile    regular verification and profiling (default)
                                                    --mode=dry_run    no kernels are launched or workspaces allocated
                                                    --mode=enumerate  lists all operation kind and operations
                                                    --mode=trace      executes a single device-side computation with
                                                                       no other kernel launches

  --device-info                                    Prints information on all GPUs present in the system

  --operation=<operation_kind>                     CUTLASS operation to profile.

  --kernels=<string_list>                          Filter operations by kernel names. For example, call all kernels with
                                                   ("s1688" and "nt") or ("s844" and "tn" and "align8") in their
                                                   operation name using --kernels="s1688*nt, s884*tn*align8"

  --ignore-kernels=<string_list>                   Excludes kernels whose names match anything in this list.

Device:
  --device=<int>                                   CUDA Device ID

  --compute-capability=<int>                       Override the compute capability.

  --llc-capacity=<capacity in KiB>                 Capacity of last-level cache in kilobytes. If this is non-zero,
                                                   profiling phases cycle through different input tensors to induce
                                                   capacity misses in the L2.


Initialization:
  --initialization=<bool>                          Enables initialization (default: true). If false, device memory is
                                                   not initialized after allocation.

  --initialization-provider=<provider>             Selects initialization provider {host, device*}. (default: '*')

  --dist=<distribution>                            Data distribution of input tensors {uniform*, gaussian, identity, sequential}
                                                    --dist=uniform,min:<double>,max:<double>,scale:<integer>
                                                    --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>
                                                    --dist=sequential,start:<double>,delta:<double>,scale:<integer>
                                                    --dist=identity

  --seed=<int>                                     Random number generator seed. Used to enforce deterministic
                                                   initialization.


Library:
  --library-algo-mode=<mode>                       Indicates algorithm mode used to call libraries such as cuBLAS and cuDNN.
                                                   mode={default*,matching,best}

  --library-algos=<range-list>                     If --algorithm-mode=best, permits specifying a selection of algorithms.


Profiling:
  --workspace-count=<workspace count>              Number of discrete workspaces maintained to avoid cache-resident 
                                                 If zero (default), the amount is chosen for each workload based on 
                                                 capacity of the last-level cache.

  --profiling-iterations=<iterations>              Number of iterations to profile each kernel. If zero, kernels
                                                   are launched up to the profiling duration.

  --warmup-iterations=<iterations>                 Number of iterations to execute each kernel prior to profiling.

  --sleep-duration=<duration>                      Number of ms to sleep between profiling periods (ms).

  --profiling-enabled=<bool>                       If true, profiling is actually conducted.

Verification:
  --verification-enabled=<bool>                    Whether to perform verification checks.

  --epsilon=<error>                                Error threshold. Setting to zero (default) requires
                                                   bit-level equivalence.

  --nonzero-floor=<floor>                          Results whose absolute value is less than this quantity
                                                   are treated as zero for comparisons.

  --save-workspace=<string>                        Specifies when to save the GEMM inputs and results to the filesystem.
                                                    --save-workspace=never      never save workspace (default)
                                                    --save-workspace=incorrect  save workspace for incorrect results
                                                    --save-workspace=always     always save workspace

  --verification-providers=<providers>             List of providers used to verify result. (default: '*')
                                                   Gemm verification-providers {cublas*}
                                                   Conv2d verification-providers {cudnn*, device*, host}


Report:
  --append=<bool>                                  If true, result is appended to possibly existing file. Otherwise, 
                                                   any existing file is overwritten.

  --output=<path>                                  Path to output file for machine readable results. Operation kind and '.csv' is appended.

  --junit-output=<path>                            Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.

  --report-not-run=<bool>                          If true, reports the status of all kernels including those that
                                                   do not satisfy the given arguments.

  --tags=<column:tag,...>                          Inserts leading columns in output table and uniform values for each
                                                   column. Useful for generating pivot tables.

  --verbose=<bool>                                 Prints human-readable text to stdout. If false, nothing is written to stdout.


About:
  --version                                        CUTLASS 2.4.0 built on Nov 19 2020 at 11:59:00


Operations:

     gemm                                          General matrix-matrix product. D = alpha * A*B + beta * C
     spgemm                                        Structured sparse GEMM. D = alpha * A*B + beta * C
     conv2d                                        Conv2d operation. Output(Tensor4D) = alpha * Input(Tensor4D) * Filter(Tensor4D) + beta * Input(Tensor4D)
     conv3d                                        Conv3d operation. Output(Tensor5D) = alpha * Input(Tensor5D) * Filter(Tensor5D) + beta * Input(Tensor5D)


For details about a particular function, specify the function name with --help.

Example:

  $ cutlass_profiler --operation=Gemm --help

  $ cutlass_profiler --operation=Conv3d --help

  $ cutlass_profiler --operation=Conv2d --help

```

# GEMM

The CUTLASS Profiler is capable of executing GEMM and Sparse GEMM problems.

The CUTLASS Profiler can be built with cuBLAS enabled to use as a reference implementation. If CMake detects
the cuBLASS library available in the system, it is included as a dependency. This may be explicitly overridden
with CMake flag `CUTLASS_ENABLE_CUBLAS`. 

## GEMM Arguments

The complete set of arguments available to each operation may be viewed by specifying the operation name
in addition to `--help`. The argument flags and their aliases usable for GEMM appear as follows.

```bash
$ ./tools/profiler/cutlass_profiler --operation=gemm --help

GEMM

  [enum]      --Gemm_kind                                       Variant of GEMM (e.g. gemm, batched, ...)
  [int]       --m,--problem-size::m                             M dimension of the GEMM problem space
  [int]       --n,--problem-size::n                             N dimension of the GEMM problem space
  [int]       --k,--problem-size::k                             K dimension of the GEMM problem space
  [tensor]    --A                                               Tensor storing the A operand
  [tensor]    --B                                               Tensor storing the B operand
  [tensor]    --C                                               Tensor storing the C operand
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [int]       --split_k_slices                                  Number of partitions of K dimension
  [int]       --batch_count                                     Number of GEMMs computed in one batch
  [enum]      --op_class,--opcode-class                         Class of math instruction (SIMT or TensorOp).
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type.
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension.
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension.
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension.
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply.
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension.
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension.
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension.
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension.
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension.
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension.
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability.
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability.

Examples:

Profile a particular problem size:
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128

Schmoo over problem size and beta:
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2

Schmoo over accumulator types:
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --accumulator-type=f16,f32

Run when A is f16 with column-major and B is any datatype with row-major 
(For column major, use column, col, or n. For row major use, row or t):

  $ ./tools/profiler/cutlass_profiler --operation=Gemm --A=f16:column --B=*:row

Using various input value distribution:
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3
  $ ./tools/profiler/cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1

Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect 
(note that --cta-tile::k=32 is default cta-tile size):
 $ ./tools/profiler/cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect

Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:
 $ ./tools/profiler/cutlass_profiler  --operation=Gemm \
   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --k=8,16,32,64,128,256,288,384,504,512,520 \
   --beta=0,1,2 --profiling-iterations=1 \
   --output=functional-test.csv
```

## Example CUDA Core GEMM Operation

Example command line for profiling SGEMM kernels is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s
```

Note, the arguments which appear in the output may be used as command line parameters for subsequent invocations.


## Example Tensor Core GEMM Operations

To execute kernels targeting Tensor Core operations, supply the flag `--op_class=tensorop` in the command line.
```bash
$ ./tools/profiler/cutlass_profiler --op_class=tensorop --m=3456 --n=4096 --k=8192



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s16816gemm_f16_256x128_32x3_nn_align8

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=8192 --A=f16:column --B=f16:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128 --cta_k=32 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 231956545536  flops

         Runtime: 0.98647  ms
          Memory: 170.272 GiB/s

            Math: 235138 GFLOP/s
```

## Covering the problem space

All arguments may have single values or comma-delimited set of values. Integers may also be specified
as an inclusive range with the following syntax `start:end:increment` or simply `start:end`. 

For example, the following sweeps over the range of the GEMM K dimension from 8 to 4096 in increments
of 8 elements.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn --m=4352 --n=4096 --k=8:4096:8
```

## Output

By default, runtime and computed GFLOP/s are reported for each operation and problem size. Additionally,
a table of comma separated values are reported at the end of the execution. This may be output to a file
with the `--output=<filename.csv>` command line option as shown:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv
```

To faclitate generation of pivot tables and charts, additional columns may be prepended with the
`--tags=<column>:<value>` option. One or more tags may be specified using a comma-delimited list.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv \
                                    --tags=cutlass:2.2,date:2020-06-08
```  

# Convolution

The CUTLASS Profiler is capable of executing 2-D and 3-D convolution problems for forwards and backwards
oeprator variants.

The CUTLASS Profiler can be built with cuDNN enabled to use as a reference implementation. If CMake detects
the cuDNN library available in the system, it is included as a dependency. This may be explicitly overridden
with CMake flag `CUTLASS_ENABLE_CUDNN`. 

```bash
$ cmake .. -DCUTLASS_LIBRARY_OPERATIONS=conv2d -DCUTLASS_ENABLE_CUDNN=OFF
...
$ make -j16 cutlass_profiler
```


## Convolution Arguments

```bash
$ ./tools/profiler/cutlass_profiler --help --operation=Conv2d

Conv2d

  [enum]      --conv_kind                                       Convolutional operator (fprop, dgrad, wgrad)
  [int]       --n,--input_n                                     Input N dimension of the Conv2d problem space
  [int]       --h,--input_h                                     Input H dimension of the Conv2d problem space
  [int]       --w,--input_w                                     Input W dimension of the Conv2d problem space
  [int]       --c,--input_c                                     Input C dimension of the Conv2d problem space
  [int]       --k,--filter_k                                    Filter K dimension of the Conv2d problem space
  [int]       --r,--filter_r                                    Filter R dimension of the Conv2d problem space
  [int]       --s,--filter_s                                    Filter S dimension of the Conv2d problem space
  [int]       --p,--output_p                                    Output P dimension of the Conv2d problem space
  [int]       --q,--output_q                                    Output Q dimension of the Conv2d problem space
  [int]       --pad_h                                           Padding in H direction
  [int]       --pad_w                                           Padding in W direction
  [int]       --stride_h                                        Stride in H direction
  [int]       --stride_w                                        Stride in W direction
  [int]       --dilation_h                                      Dilation in H direction
  [int]       --dilation_w                                      Dilation in W direction
  [tensor]    --Activation                                      Tensor storing the Activation operand
  [tensor]    --Filter                                          Tensor storing the Filter operand
  [tensor]    --Output                                          Tensor storing the Output operand
  [enum]      --conv_mode                                       Convolution filter mode (conv, cross)
  [enum]      --iterator_algorithm,--iterator_algo              Convolution iterator algorithm (analytic, optimized)
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     SplitK mode for serial or parallel reduction (serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [enum]      --eq_gemm_provider,--eq-gemm-provider             Enable profiling equivalent gemm by the following providers (cutlass)
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma)
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability

Examples:

Profile a particular convolution (specify all the convolution parameters):

 $ cutlass_profiler --operation=Conv2d --Activation=f16:nhwc   \
  --Filter=f16:nhwc --Output=f16 --accumulator-type=f32        \
  --n=32 --h=14 --w=14 --c=8 --k=64 --r=3 --s=3                \
  --pad_h=1 --pad_w=1                                          \
  --stride::h=1 --stride::w=1 --dilation::h=1 --dilation::w=1

```

## Example CUDA Core Convolution Operation

Example command line for profiling forward propagation convolution kernels on CUDA cores is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=simt_sfprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_simt_sfprop_optimized_128x128_8x2_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops

         Runtime: 8.13237  ms
          Memory: 235.431 GiB/s

            Math: 14569.3 GFLOP/s

```

## Example Tensor Core Convolution Operation

Example command line for profiling forward propagation convolution kernels runing on Tensor Cores is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=tensorop*fprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_64x4_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=64 --stages=4  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.945071  ms
          Memory: 1114.21 GiB/s

            Math: 125369 GFLOP/s


```

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
