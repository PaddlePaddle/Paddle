![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")

# CUTLASS 2.9

_CUTLASS 2.9 - April 2022_

CUTLASS is a collection of CUDA C++ template abstractions for implementing
high-performance matrix-multiplication (GEMM) and related computations at all levels 
and scales within CUDA. It incorporates strategies for hierarchical decomposition and 
data movement similar to those used to implement cuBLAS and cuDNN.  CUTLASS decomposes 
these "moving parts" into reusable, modular software components abstracted by C++ template 
classes.  These thread-wide, warp-wide, block-wide, and device-wide primitives can be specialized
and tuned via custom tiling sizes, data types, and other algorithmic policy. The
resulting flexibility simplifies their use as building blocks within custom kernels
and applications.

To support a wide variety of applications, CUTLASS provides extensive support for
mixed-precision computations, providing specialized data-movement and
multiply-accumulate abstractions for half-precision floating
point (FP16), BFloat16 (BF16), Tensor Float 32 (TF32),
single-precision floating point (FP32), double-precision floating
point (FP64) types, integer data types (4b and 8b), and binary data types (1b). 
CUTLASS demonstrates warp-synchronous matrix multiply operations 
targeting the  programmable, high-throughput _Tensor Cores_ implemented by 
NVIDIA's Volta, Turing, and Ampere architectures.

CUTLASS implements high-performance Convolution via the implicit GEMM algorithm.
Implicit GEMM is the formulation of a convolution operation as a GEMM thereby taking advantage of
CUTLASS's modular GEMM pipeline. 
This allows CUTLASS to build convolutions by reusing highly optimized warp-wide GEMM components and below. 

See the [Quick Start Guide](/media/docs/quickstart.md) to get started quickly.

See the [functionality listing](/media/docs/functionality.md) for the list of operations
supported at each level of the execution model hierarchy.

# What's New in CUTLASS 2.9

CUTLASS 2.9 is an update to CUTLASS adding:
- [First layer Convolution kernels](/test/unit/conv/device/conv2d_fprop_fixed_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) specialized for small channel counts and reduced alignment
- [BLAS3](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference) operators accelerated by Tensor Cores
  - [SYRK](/test/unit/gemm/device/syrk_f32n_f32t_tensor_op_fast_f32_sm80.cu), [HERK](/test/unit/gemm/device/herk_cf32h_cf32n_tensor_op_fast_f32_sm80.cu),
  - [SYR2K](/test/unit/gemm/device/syr2k_f32n_f32n_tensor_op_fast_f32_sm80.cu), [HER2K](/test/unit/gemm/device/her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cu),
  - [Out-of-place TRMM](/test/unit/gemm/device/trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu), and 
  - [SYMM](/test/unit/gemm/device/symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu), [HEMM](/test/unit/gemm/device/hemm_cf32h_cf32n_tensor_op_fast_f32_ls_sm80.cu)
- [CUTLASS Python](/examples/40_cutlass_py) demonstrating JIT compilation of CUTLASS kernels and a Python-based runtime using [CUDA Python](https://developer.nvidia.com/cuda-python)
- [GEMM + Softmax example](/examples/35_gemm_softmax)
- [Gather and Scatter Fusion with GEMM](/examples/36_gather_scatter_fusion) can gather inputs and scatters outputs based on indices vectors in the same GEMM kernel.
- [Back-to-back GEMM/CONV](examples/13_two_tensor_op_fusion) fully supports buffering the first GEMM/CONV results in the shared memory for the latter one to use.  Bias Vector add is also supported in the first GEMM/CONV.
- [Transposed Convolution](/examples/34_transposed_conv2d) (a.k.a Deconvolution) support which reuses Dgrad implementation.
- [Utility functions](/tools/util/include/cutlass/util) that can pad NHWC and convert between NCHW and NHWC.
- [Small alignment implicit gemm](https://github.com/NVIDIA/cutlass/issues/242) support for Fprop/Dgrad/Wgrad so that padding is no longer mandated to use tensor cores.
- Epilogue enhancement with performance improvement, more activation functions, and more fusion patterns.
- [Group GEMM](/examples/24_gemm_grouped) thread block number calculation fix.
- Optimal performance using [CUDA 11.7](https://developer.nvidia.com/cuda-downloads)
- [Parallel GEMM splitk](https://github.com/NVIDIA/cutlass/pull/277) support in the CUTLASS profiler.
- Updates and bugfixes from the community (thanks!)
- **Deprecation announcement:** CUTLASS plans to deprecate the following:
  - Maxwell and Pascal GPU architectures
  - Ubuntu 16.04
  - CUDA 10.2

**See the [CHANGELOG](CHANGELOG.md) for a detailed listing of releases and updates.**

# Performance

<p align="center"><img src=/media/images/cutlass-2.8-gemm-performance.png></p>

CUTLASS primitives are very efficient.  When used to construct device-wide GEMM kernels,
they exhibit performance comparable to cuBLAS for scalar GEMM
computations. The above figure shows CUTLASS performance relative to cuBLAS
for large matrix dimensions on an [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/), 
an [NVIDIA A2](https://www.nvidia.com/en-us/data-center/products/a2/), 
an [NVIDIA TitanV](https://www.nvidia.com/en-us/titan/titan-v/), 
and an [NVIDIA GeForce 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/)
compiled with the [CUDA 11.5 Toolkit](https://developer.nvidia.com/cuda-downloads). Tensor Core operations are implemented using CUDA's 
[mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

<p align="center"><img src=/media/images/cutlass-2.9-implicit-gemm-performance.png></p>

When using CUTLASS building blocks to construct device-wide implicit gemm (Fprop, Dgrad, and Wgrad)
kernels, CUTLASS performance is also comparable to cuDNN when running Resnet-50 layers on an [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)
as shown in the above figure.  Tensor Core operations are still implemented using CUDA's
[mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

# Compatibility

CUTLASS requires a C++11 host compiler and 
performs best when built with the [**CUDA 11.6u2 Toolkit**](https://developer.nvidia.com/cuda-toolkit).
It is also compatible with CUDA 11.0, CUDA 11.1, CUDA 11.2, CUDA 11.3, CUDA 11.4, and CUDA 11.5.

We have tested the following environments.

|**Operating System** | **Compiler** |
|-----------------|----------|
| Windows 10      | Microsoft Visual Studio 2015|
|                 | Microsoft Visual Studio 2017|
|                 | Microsoft Visual Studio 2019|
| Ubuntu 18.04 | GCC 7.5.0 |
| Ubuntu 20.04 | GCC 10.3.0 |
| Ubuntu 21.04 | GCC 11.2.0 |

Additionally, CUTLASS may be built with clang. 
See [these instructions](media/docs/quickstart.md#clang) for more details.

CUTLASS runs successfully on the following NVIDIA GPUs, and it is expected to be efficient on
any Volta-, Turing-, or NVIDIA Ampere- architecture NVIDIA GPU. 

|**GPU**|**CUDA Compute Capability**|**Minimum CUDA Toolkit**|**Minimum CUDA Toolkit Enabling Native Tensor Cores**|
|---|---|---|---|
|NVIDIA Tesla V100|7.0|9.2|10.1|
|NVIDIA TitanV|7.0|9.2|10.1|
|NVIDIA GeForce RTX 2080 TI, 2080, 2070|7.5|10.0|10.2|
|NVIDIA Tesla T4|7.5|10.0|10.2|
|NVIDIA A100|8.0|11.0|11.0|
|NVIDIA A10 |8.6|11.1|11.1|
|NVIDIA GeForce 3090|8.6|11.1|11.1|

For all GPUs, we recommend compiling with the [CUDA 11.6u2 Toolkit](https://developer.nvidia.com/cuda-toolkit)
for best performance.

# Documentation

CUTLASS is described in the following documents and the accompanying
[Doxygen documentation](https://nvidia.github.io/cutlass).

- [Quick Start Guide](/media/docs/quickstart.md) - build and run CUTLASS
- [Functionality](/media/docs/functionality.md) - summarizes functionality available in CUTLASS
- [Efficient GEMM in CUDA](media/docs/efficient_gemm.md) - describes how GEMM kernels may be implemented efficiently in CUDA
- [GEMM API](media/docs/gemm_api.md) - describes the CUTLASS GEMM model and C++ template concepts 
- [Implicit GEMM Convolution](media/docs/implicit_gemm_convolution.md) - describes 2-D and 3-D convolution in CUTLASS
- [Code Organization](media/docs/code_organization.md) - describes the organization and contents of the CUTLASS project
- [Terminology](media/docs/terminology.md) - describes terms used in the code
- [Programming Guidelines](media/docs/programming_guidelines.md) - guidelines for writing efficient modern CUDA C++
- [Fundamental types](media/docs/fundamental_types.md) - describes basic C++ classes used in CUTLASS to represent numeric quantities and arrays
- [Layouts](media/docs/layout.md) - describes layouts of matrices and tensors in memory
- [Tile Iterators](media/docs/tile_iterator_concept.md) - describes C++ concepts for iterating over tiles of matrices in memory
- [CUTLASS Profiler](media/docs/profiler.md) - command-line driven profiling application
- [CUTLASS Utilities](media/docs/utilities.md) - additional templates used to facilate rapid development

We have also described the structure of an efficient GEMM in our talk at the
[GPU Technology Conference 2018](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf).

# Building CUTLASS

CUTLASS is a header-only template library and does not need to be built to be used by other
projects. Client applications should target CUTLASS's `include/` directory in their include
paths.

CUTLASS unit tests, examples, and utilities can be build with CMake starting version 3.12. 
Make sure the `CUDACXX` environment  variable points to NVCC in the CUDA Toolkit installed
on your system.

```bash
$ export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
```

Create a build directory within the CUTLASS project, then run CMake. By default CUTLASS will build kernels
for CUDA architecture versions 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, and 8.6. To reduce compile time you can specify
the architectures to build CUTLASS for by changing the CMake configuration setting
`CUTLASS_NVCC_ARCHS`.

```bash
$ mkdir build && cd build

$ cmake .. -DCUTLASS_NVCC_ARCHS=80               # compiles for NVIDIA's Ampere Architecture
```

From the `build/` directory, compile and run the CUTLASS unit tests by building the target `test_unit` with make.

The unit tests are organized as several binaries mirroring the top-level namespaces of CUTLASS,
and they may be executed in parallel via make's `-j` command line argument.

```bash
$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] 946 tests from 57 test cases ran. (10812 ms total)
[  PASSED  ] 946 tests.
```

All tests should pass on supported platforms, though the exact number of tests may vary over time.


# Project Structure

CUTLASS is arranged as a header-only library along with Utilities, Tools, Examples, and unit tests. 
[Doxygen documentation](https://nvidia.github.io/cutlass) provides a complete list of files, classes, 
and template concepts defined in the CUTLASS project.

A detailed explanation of the source code organization may be found in the 
[CUTLASS documentation](media/docs/code_organization.md), but several main components are summarized below.

## CUTLASS Template Library

```
include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution

    gemm/                    # code specialized for general matrix product computations

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # CUDA-capable Standard Library components

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" model
    
    transform/               # code specialized for layout, type, and domain transformations

    *                        # core vocabulary types, containers, and basic numeric operations
```

### CUTLASS SDK Examples

[CUTLASS SDK examples](/examples) apply CUTLASS templates to implement basic computations.

```
examples/
  00_basic_gemm/                   # launches a basic GEMM with single precision inputs and outputs

  01_cutlass_utilities/            # demonstrates CUTLASS Utilities for allocating and initializing tensors
  
  02_dump_reg_smem/                # debugging utilities for printing register and shared memory contents
  
  03_visualize_layout/             # utility for visualizing all layout functions in CUTLASS

  04_tile_iterator/                # example demonstrating an iterator over tiles in memory

  05_batched_gemm/                 # example demonstrating CUTLASS's batched strided GEMM operation

  06_splitK_gemm/                  # exmaple demonstrating CUTLASS's Split-K parallel reduction kernel

  07_volta_tensorop_gemm/          # example demonstrating mixed precision GEMM using Volta Tensor Cores

  08_turing_tensorop_gemm/         # example demonstrating integer GEMM using Turing Tensor Cores

  09_turing_tensorop_conv2dfprop/  # example demonstrating integer implicit GEMM convolution (forward propagation) using Turing Tensor Cores

  10_planar_complex/               # example demonstrating planar complex GEMM kernels

  11_planar_complex_array/         # example demonstrating planar complex kernels with batch-specific problem sizes

  12_gemm_bias_relu/               # example demonstrating GEMM fused with bias and relu

  13_fused_two_gemms/              # example demonstrating two GEMms fused in one kernel

  22_ampere_tensorop_conv2dfprop/  # example demonstrating integer implicit GEMM convolution (forward propagation) using Ampere Tensor Cores

  31_basic_syrk                    # example demonstrating Symetric rank-K update

  32_basic_trmm                    #

  33_ampere_3xtf32_tensorop_symm   #

  35_gemm_softmax                  # example demonstrating GEMM fused with Softmax in mixed precision using Ampere Tensor Cores

  40_cutlass_py                    # example demonstrating CUTLASS with CUDA Python
```

### Tools

```
tools/
  library/                   # CUTLASS Instance Library - contains instantiations of all supported CUTLASS templates
    include/
      cutlass/
        library/

  profiler/                  # CUTLASS Profiler         - command-line utility for executing operations in the
                             #                            CUTLASS Library
  
  util/                      # CUTLASS Utilities        - contains numerous helper classes for
    include/                 #                            manging tensors in device memory, reference
      cutlass/               #                            implementations for GEMM, random initialization
        util/                #                            of tensors, and I/O.
```

### Test

The `test/unit/` directory consist of unit tests implemented with Google Test that demonstrate
basic usage of Core API components and complete tests of the CUTLASS GEMM computations.

Instructions for building and running the Unit tests are described in the [Quickstart guide](media/docs/quickstart.md).

# Performance Profiling

The `tools/profiler/` directory contains a command-line utility for launching each of the GEMM kernels.
It can be built as follows:

```bash
$ make cutlass_profiler -j16
```
## Building all GEMM and Convolution kernels (_long_ build times)

By default, only one tile size is instantiated for each data type, math instruction, and layout.
To instantiate all, set the following environment variable when running CMake from an empty `build/` directory.
Beware, this results in *thousands* of kernels and long build times.
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j16
```

## Building a subset of GEMM and Convolution kernels (_reduced_ build times)

To compile strictly one kernel or a small set of kernels, a comma-delimited list of kernel names with 
wildcard characters may be used to reduce the set of kernels. The following examples show building exactly one
or a subset of kernels for NVIDIA Ampere and Turing architecture:

### Building a subset Tensor Core GEMM kernels

To compile a subset of Tensor Core GEMM kernels with FP32 accumulation and FP16 input targetting NVIDIA Ampere and Turing architecture, 
use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
...
$ make cutlass_profiler -j16
```

Example command line for profiling a subset of Tensor Core GEMM kernels is as follows:
```bash
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align8 --m=3456 --n=4096 --k=4096

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Passed

       Arguments: --gemm_kind=universal --m=3456 --n=4096 --k=4096 --A=f16:column --B=f16:row --C=f32:column --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128  \
                  --cta_k=32 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=8 --min_cc=75  \
                  --max_cc=1024

           Bytes: 118489088  bytes
           FLOPs: 115992428544  flops

         Runtime: 1.55948  ms
          Memory: 70.7616 GiB/s

            Math: 74378.8 GFLOP/s



=============================
...
```

### Building one CUDA Core GEMM kernel

To compile one SGEMM kernel targetting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sgemm_128x128_8x2_nn_align1
...
$ make cutlass_profiler -j16
```

Example command line for profiling single SGEMM CUDA kernel is as follows:
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

=============================
```

### Building a subset of Tensor Core Convolution kernels

To compile a subset of Tensor core convolution kernels implementing forward propagation (fprop) with FP32 accumulation 
and FP16 input targetting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*fprop_optimized_f16
...
$ make cutlass_profiler -j16
```

Example command line for profiling a subset of Tensor Core convolution kernels is as follows:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

...
=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_32x5_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=32 --stages=5  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.711496  ms
          Memory: 1479.99 GiB/s

            Math: 166526 GFLOP/s

=============================
...
```


### Building one Convolution CUDA kernel

To compile and run one CUDA Core convolution kernel implementing forward propagation (fprop) with F32 accumulation 
and FP32 input targetting NVIDIA Ampere and Turing architecture, use the below cmake command line:
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS='75;80' -DCUTLASS_LIBRARY_KERNELS=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc
...
$ make cutlass_profiler -j16
```

Example command line for profiling one CUDA Core convolution kernel:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sfprop_optimized_128x128_8x2_nhwc --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


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

         Runtime: 7.34266  ms
          Memory: 260.752 GiB/s

            Math: 16136.2 GFLOP/s


=============================

```

## More Details on Compiling CUTLASS Kernels and CUTLASS Profiler
- Please follow the links for more CMake examples on selectively compiling CUTLASS kernels:
  - [GEMM CMake Examples](media/docs/quickstart.md#gemm-cmake-examples) 
  - [Implicit GEMM conovlution CMake Examples](media/docs/quickstart.md#convolution-cmake-examples)
- [Further details about the CUTLASS Profiler are described here.](media/docs/profiler.md)


# About

CUTLASS is released by NVIDIA Corporation as Open Source software under the 
[3-clause "New" BSD license](LICENSE.txt).

# Contributors

The official list of CUTLASS developers and contributors is available here: [CONTRIBUTORS](CONTRIBUTORS.md).

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

