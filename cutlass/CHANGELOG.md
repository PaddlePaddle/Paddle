# NVIDIA CUTLASS Changelog

## [2.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.9.0) (2022-04-21)

* [First layer Convolution kernels](/test/unit/conv/device/conv2d_fprop_fixed_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) specialized for small channel counts and reduced alignment
  * [Few channels](/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h) specialization for reduced alignment capabilities
  * [Fixed channels](/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h) further specialized when channel count perfectly matches the access vector size
  * [Unit tests](/test/unit/conv/device/conv2d_fprop_few_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu)
  * [Python-based instance emitter](/tools/library/scripts/generator.py) in the CUTLASS Library and support in the Profiler
* [BLAS3](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference) operators accelerated by Tensor Cores
  * Supported types: f32, cf32, f64, cf64, tf32x3, complex tf32x3
  * [HERK](/test/unit/gemm/device/her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cu) with [emitter](/tools/library/scripts/rank_k_operation.py)
  * [SYRK](/test/unit/gemm/device/syrk_f32n_f32t_tensor_op_fast_f32_sm80.cu) with [emitter](/tools/library/scripts/rank_k_operation.py)
  * [SYMM](/test/unit/gemm/device/symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu) with [emitter](/tools/library/scripts/symm_operation.py)
  * [TRMM](/test/unit/gemm/device/trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu) with [emitter](/tools/library/scripts/trmm_operation.py)
  * [Unit tests](/test/unit/gemm/device/testbed_rank_k_universal.h)
* [CUTLASS Python](/examples/40_cutlass_py) demonstrating JIT compilation of CUTLASS kernels and a Python-based runtime using [CUDA Python](https://developer.nvidia.com/cuda-python)
  * [Python-based runtime](/tools/library/scripts/rt.py) interoperable with existing emitters
* [GEMM + Softmax example](/examples/35_gemm_softmax)
* [Gather and Scatter Fusion with GEMM](/examples/36_gather_scatter_fusion) can gather inputs and scatters outputs based on indices vectors in the same GEMM kernel.
  * It can select random rows in a row major matrix.
  * It can select random columns in a column major matrix.
* [Back-to-back GEMM/CONV](examples/13_two_tensor_op_fusion) fully supports buffering the first GEMM/CONV results in the shared memory for the latter one to use.  It can eliminate register spill when the tile size is big.  Additionally, bias vector add is supported in the first GEMM/CONV.
  * Supported kernels: GEMM and CONV.
  * Supported types: fp16 and int8.
  * Supported architectures: Turing and Ampere.
* [Transposed Convolution](/examples/34_transposed_conv2d) (a.k.a Deconvolution) support which reuses Dgrad implementation.
* [Utility functions](/tools/util/include/cutlass/util) that can pad NHWC and convert between NCHW and NHWC.
* [Small alignment implicit gemm](https://github.com/NVIDIA/cutlass/issues/242) support for Fprop/Dgrad/Wgrad so that padding is no longer mandated to use tensor cores in these kernels.
* Epilogue enhancement:
  * Eliminate bank conflicts in int8 tensor core kernels.
  * Half2 usage if epilogue compute type is fp16.
  * More activation functions: Silu, Hardswish, Leaky Relu.
  * New elementwise fusion pattern for [residual block](/include/cutlass/epilogue/thread/linear_combination_residual_block.h).
* [Group GEMM](/examples/24_gemm_grouped) thread block number calculation fix which helps to launch the intended number of threadblocks to fully occupy the GPUs.
* [Parallel GEMM splitk](https://github.com/NVIDIA/cutlass/pull/277) support in the CUTLASS profiler.
* Optimal performance using [**CUDA 11.7**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [2.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.8.0) (2021-11-19)

* **TF32x3:** emulated single-precision using Tensor Cores
  * 45+ TFLOPs on NVIDIA A100
  * [GEMM SDK example](/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu) (real)
  * [COMPLEX GEMM SDK example](/examples/29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm/29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm.cu) (complex)
  * [Implicit GEMM Convolution SDK example](/examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/ampere_3xtf32_fast_accurate_tensorop_fprop.cu)
* **Mainloop fusion for Convolution:** convolution with fused per-channel scale-bias-relu
  * [Conv Fprop SDK example](/examples/25_ampere_fprop_mainloop_fusion/ampere_fprop_mainloop_fusion.cu)
  * [Conv WGrad SDK example](/examples/26_ampere_wgrad_mainloop_fusion/ampere_wgrad_mainloop_fusion.cu) 
  * [cutlass::conv::device::ImplicitGemmConvolutionFusion](/include/cutlass/conv/device/implicit_gemm_convolution_fusion.h)
* **Grouped GEMM:** similar to batched GEMM with distinct problem size per group
  * [SDK example](/examples/24_gemm_grouped) with performance comparison with Batched Strided GEMM
  * [cutlass::gemm::device::GemmGrouped](/include/cutlass/gemm/device/gemm_grouped.h)
* [Implicit GEMM Convolution fusion](/examples/13_two_tensor_op_fusion/) supports staging 1st convolution's output accumulator in the shared memory on Turing. This allows more flexible warp tile sizes and less regsiter pressue.
* Optimal performance using [**CUDA 11.5**](https://developer.nvidia.com/cuda-downloads)
* Updates from the community (thanks!)

* **Deprecation announcement:** CUTLASS plans to deprecate the following:
  * Maxwell and Pascal GPU architectures
  * Ubuntu 16.04
  * CUDA 10.2

## [2.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.7.0) (2021-09-24)
  * Mainloop fusion for GEMM: [summation over A or B](/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu)
  * [Strided DGRAD (optimized iterators)](/include/cutlass/conv/kernel/default_conv2d_dgrad.h)
  * [Half-precision GELU_taylor activation functions](/include/cutlass/epilogue/thread/activation.h#L196)
    * Use these when accumulation and epilogue compute types are all `cutlass::half_t`
  * Tuning and bug fixes to [fused GEMM + GEMM example](/examples/13_two_tensor_op_fusion/)
  * Support for smaller than 128b aligned Convolutions: [see examples](test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu#L272)
  * Caching of results to accelerate Convolution [unit tests](test/unit/conv/device/cache_testbed_output.h)
    * Can be enabled or disabled by running `cmake .. -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=OFF`
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.1](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.1) (2021-09-03)
  * Arbitrary padding and striding for CUTLASS Strided DGRAD Convolution operator (Analytic Iterators)
  * Tuning for GEMMs fused with partial reductions
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.0) (2021-07-22)
  * Optimal performance when compiled with the [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit)
    * Adopt the new L2 prefetch feature in [cp.async](/include/cutlass/arch/memory.h) and [global load](/include/cutlass/arch/memory_sm80.h)
  * Fused operators with GEMM and Convolution
    * [Fused broadcast in epilogue](test/unit/gemm/device/gemm_with_broadcast_f16n_f16n_f16n_tensorop_f32_sm75.cu)
    * [Fused partial reduction in epilogue](/test/unit/gemm/device/gemm_with_reduction_f16n_f16n_f16n_tensorop_f32_sm75.cu)
  * 64b tensor strides and leading dimensions support for GEMMs
  * Affine rank=2 matrix layouts 
    * Row stride and column stride for matrices using [cutlass::layout::AffineRank2](/include/cutlass/layout/matrix.h)
    * Support [FP64 tensor core](/examples/18_ampere_fp64_tensorop_affine2_gemm/ampere_fp64_tensorop_affine2_gemm.cu) and SIMT GEMM.
  * [Batched GEMV](/test/unit/gemm/device/gemv.cu) preview implementation
  * [New strided Dgrad](test/unit/conv/device/conv2d_strided_dgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) implementation
    * Accelerates over previous implementation by cutting down redundant math by 4x
    * Support using new `Dy` and `w` analytic iterators and existing `cutlass::conv::device::ImplicitGemmConvolution` interface
  * Quaternion-valued GEMM and Convolution in single- and double-precision (targeting CUDA Cores)
    * Updates to [quaternion.h](/include/cutlass/quaternion.h) and [functional.h](/include/cutlass/functional.h)
    * SDK Example for [GEMM](/examples/21_quaternion_gemm/quaternion_gemm.cu) and [Convolution](/examples/22_quaternion_gemm/quaternion_conv.cu)
    * [Unit tests for GEMM](/test/unit/gemm/device/simt_qgemm_nn_sm50.cu) and [Convolution](/test/unit/conv/device/conv2d_fprop_implicit_gemm_qf32nhwc_qf32nhwc_qf32nhwc_simt_f32_sm50.cu)
  * Many improvements to the epilogue.
    * Provide an [option](/include/cutlass/epilogue/threadblock/epilogue.h) to not fully unroll the epilogue to reduce the code size and improve the performance when using complicated elementwise operations
    * Performance improvement for FP16 tensor core kernels
    * Bug fixes
  * Enhanced Clang support and the combination of Clang 13 and CUDA 11.4 can build and run kernels from Pascal and Ampere.   
  * Updated minimum CUDA Toolkit requirement to 10.2
    * [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit) recommended
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.5.0) (2021-02-26)
  * Tensor reductions
    * _m_-to-_n_ reductions of tensors with affine layout
    * [Specializations](/test/unit/reduction/device/tensor_reduce_contiguous.cu) for reductions including contiguous dimension
    * [Specializations](/test/unit/reduction/device/tensor_reduce_strided.cu) for reductions excluding contiguous dimension
    * Custom reduction functors such as `cutlass::logical_and`
    * Large tensor support, up to 2^63 elements (however, each dimension is limited to an extent of 2^31)
  * Optimizations for 3-D convolution
    * [Optimized tile iterators](include/cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h) using precomputed delta table for 3-D convolution
    * Full coverage of [forward](test/unit/conv/device/conv3d_fprop_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) and [backwards](test/unit/conv/device/conv3d_dgrad_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) passes for 3D convolution
  * [Fused Convolution+Convolution example](/examples/13_two_tensor_op_fusion/README.md)
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!
  

## [2.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.4.0) (2020-11-19)
  * Implicit GEMM convolution kernels supporting CUDA and Tensor Cores on NVIDIA GPUs
    * Operators: forward (Fprop), backward data gradient (Dgrad), and backward weight gradient (Wgrad) convolution
    * Data type: FP32, complex<FP32>, Tensor Float 32 (TF32), BFloat16 (BF16), Float16, Int4, Int8, Int32
    * Spatial dimensions: 1-D, 2-D, and 3-D
    * Layout: NHWC, NCxHWx
  * Implicit GEMM convolution components: 
    * Global memory iterators supporting Fprop, Dgrad, and Wgrad
    * `MmaMultistage` for implicit GEMM convolution for NVIDIA Ampere architecture
    * `MmaPipeline` for implicit GEMM convolution for NVIDIA Volta and Turing architectures
    * [Documentation](/media/docs/implicit_gemm_convolution.md) describing Implicit GEMM Convolution algorithm and implementation

## [2.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.3.0) (2020-09-23)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * [Sparse Tensor Core GEMM kernels](test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu):
     * Direct access to Sparse Tensor Cores and maximum performance via [`mma.sp.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
   * Fast SGEMM targeting GeForce RTX 30-series CUDA Cores
 * Minor Features:
   * [Activation functions](/include/cutlass/epilogue/thread/activation.h) such as [GeLU](/include/cutlass/epilogue/thread/linear_combination_gelu.h) and [Sigmoid](/include/cutlass/epilogue/thread/linear_combination_sigmoid.h)
   * Small [matrix](/include/cutlass/matrix.h) and [quaternion](/include/cutlass/quaternion.h) template classes in device code
   * [Floating-point constants](/include/cutlass/constants.h)
 * NVIDIA Ampere GPU Architecture examples and documentation:
   * [Tensor Float 32](/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu) and 
   * [Sparse Tensor Cores](/examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm.cu)
   * Documentation added on CUTLASS [efficient row-major epilogue](/media/docs/gemm_api.md#efficient-epilogue)

## [2.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.2.0) (2020-06-08)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * Fast Tensor Core operations: 
    * Maximum performance via [`mma.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
    * Tensor Float 32, BFloat16, and double-precision data types
    * Mixed integer data types (int8, int4, bin1)
   * Asynchronous copy for deep software pipelines via [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution)   
   * Described in [GTC 2020 Webinar (SR 21745)](https://developer.nvidia.com/gtc/2020/video/s21745) (free registration required)
 * Features:
   * SDK examples showing GEMM fused with bias+relu and fused GEMM+GEMM
   * Complex-valued GEMMs targeting NVIDIA Ampere Tensor Cores in double-precision and Tensor Float 32
   * Gaussian complex GEMMs using 3m complex multiply algorithm
   * Universal GEMM kernel supporting two batch modes and two algorithms for parallel reductions
 * Policy updates:
   * [CUDA 11 Toolkit](https://developer.nvidia.com/cuda-toolkit) needed to enable NVIDIA Ampere Architecture features
   * Disabled F16C by default for compatibility - enable on cmake command line with `-DCUTLASS_ENABLE_F16C=ON`

## [2.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.1.0) (2020-04-06)
 * BLAS-style host-side API added to [CUTLASS Library](/media/docs/quickstart.md#cutlass-library)
    * API to launch compiled kernel instances for GEMM and planar complex GEMM
 * Planar Complex GEMM kernels targeting Volta and Turing Tensor Cores
    * Computes complex matrix products on matrices stored as disjoint real and imaginary parts
    * [SDK Examples of Planar Complex GEMMs](/examples/10_planar_complex/planar_complex.cu)
 * Minor enhancements and bug fixes

## [2.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.0.0) (2019-11-19)
 * Substantially refactored for
    * Better performance, particularly for native Turing Tensor Cores
    * Robust and durable templates spanning the design space
    * Encapsulated functionality embodying modern C++11 programming techniques
    * Optimized containers and data types for efficient, generic, portable device code
  * Updates to:
    * [Quick start guide](/media/docs/quickstart.md)
    * [Documentation](/README.md#documentation)
    * [Utilities](/media/docs/utilities.md)
    * [CUTLASS Profiler](/media/docs/profiler.md)
 * Native Turing Tensor Cores
    * Efficient GEMM kernels targeting Turing Tensor Cores
    * Mixed-precision floating point, 8-bit integer, 4-bit integer, and binarized operands
 * Coverage of existing CUTLASS functionality
    * GEMM kernels targeting CUDA and Tensor Cores in NVIDIA GPUs
    * Volta Tensor Cores through native mma.sync and through WMMA API
    * Optimizations such as parallel reductions, threadblock rasterization, and intra-threadblock reductions
    * Batched GEMM operations
    * Complex-valued GEMMs
 * **Note: a host compiler supporting C++11 or greater is required.**

# CUTLASS 1.x

## [1.3.2](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.2) (2019-07-09)
 * Performance improvement for Volta Tensor Cores TN and TT layouts.

## [1.3.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.1) (2019-04-09)
 * Corrected NVRTC unit tests.

## [1.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.0) (2019-03-20)
 * Efficient GEMM kernel targeting Volta Tensor Cores via `mma.sync` instruction added in CUDA 10.1.

## [1.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.2.0) (2018-10-26)
 * Parallelized reductions across threadblocks ("Split-K")
   * Improved IGEMM performance
 * Batched strided WMMA GEMMs

## [1.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.1.0) (2018-09-19)
  * Turing Features
    * WMMA GEMM targeting TensorCores - INT8, INT4, 1-bit
  * Batched Strided GEMM
  * Threadblock rasterization strategies
    * Improved performance for adverse problem sizes and data layouts
  * Extended CUTLASS Core comonents
    * Tensor views support arbitrary matrix and tensor layouts
    * Zip iterators for structuring multiple data streams
  * Enhanced CUTLASS utilities
    * Reference code for tensor operations in host and device code
    * Added HostMatrix<> for simplified matrix creation
  * Examples
    * Basic GEMM, tensor views, CUTLASS utilities, batched GEMM, WMMA GEMM

## [1.0.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.0.1) (2018-06-11)

  * Intra-threadblock reduction added for small threadblock tile sizes
    * sgemm_64x128x16, sgemm_128x128x16, sgemm_128x64x16, sgemm_128x32x16, sgemm_64x64x16, sgemm_64x32x16
    * igemm_32x32x128
  * GEMM _K_ residue handled during prologue prior to mainloop
  * Replaced Google Test copy with submodule. Use `git submodule init --recursive --update`

## [1.0.0](https://github.com/NVIDIA/cutlass/commit/2028ebe120aab22bfd0b2baf8902d4c9627eb33f) (2018-05-16)

  * Substantial rewrite to accommodate new architecture
  * Kernels: SGEMM, DGEMM, IGEMM, HGEMM, WMMA GEMM
  * Unit and performance tests

## [0.0.1](https://github.com/NVIDIA/cutlass/commit/d08ba8ac46e2fa3f745e070c390182edb56b2e91) (2017-12-04)

  * Initial release


## Copyright

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

