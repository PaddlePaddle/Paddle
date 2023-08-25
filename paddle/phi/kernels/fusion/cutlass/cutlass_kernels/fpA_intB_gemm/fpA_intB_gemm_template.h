/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma once

#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/compute_occupancy.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/epilogue_helpers.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/ft_gemm_configs.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"
namespace phi {

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_mixed_gemm_kernelLauncher(const T* A,
                                       const WeightType* B,
                                       const float* weight_scales,
                                       const T* biases,
                                       T* C,
                                       int m,
                                       int n,
                                       int k,
                                       CutlassGemmConfig gemm_config,
                                       char* workspace,
                                       size_t workspace_bytes,
                                       cudaStream_t stream,
                                       int* occupancy);
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages,
          typename Enable = void>
struct dispatch_stages {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const float* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    // VLOG(3)<<__PRETTY_FUNCTION__;
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) +
                          " with stages set to " + std::to_string(Stages);
    throw std::runtime_error("[dispatch_stages::dispatch] " + err_msg);
  }
};
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
struct dispatch_stages<T,
                       WeightType,
                       arch,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       2> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const float* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    // VLOG(3)<<__PRETTY_FUNCTION__;

    generic_mixed_gemm_kernelLauncher<T,
                                      WeightType,
                                      arch,
                                      EpilogueTag,
                                      ThreadblockShape,
                                      WarpShape,
                                      2>(A,
                                         B,
                                         weight_scales,
                                         biases,
                                         C,
                                         m,
                                         n,
                                         k,
                                         gemm_config,
                                         workspace,
                                         workspace_bytes,
                                         stream,
                                         occupancy);
  }
};

template <typename T,
          typename WeightType,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
struct dispatch_stages<T,
                       WeightType,
                       cutlass::arch::Sm80,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const float* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_mixed_gemm_kernelLauncher<T,
                                      WeightType,
                                      cutlass::arch::Sm80,
                                      EpilogueTag,
                                      ThreadblockShape,
                                      WarpShape,
                                      Stages>(A,
                                              B,
                                              weight_scales,
                                              biases,
                                              C,
                                              m,
                                              n,
                                              k,
                                              gemm_config,
                                              workspace,
                                              workspace_bytes,
                                              stream,
                                              occupancy);
  }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape32x128x64_WarpShape32x32x64(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape64x128x64_WarpShape64x32x64(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape128x128x64_WarpShape128x32x64(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape128x256x64_WarpShape64x64x64(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape256x128x64_WarpShape64x64x64(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_to_cutlass(const T* A,
                              const WeightType* B,
                              const float* weight_scales,
                              const T* biases,
                              T* C,
                              int m,
                              int n,
                              int k,
                              char* workspace,
                              size_t workspace_bytes,
                              CutlassGemmConfig gemm_config,
                              cudaStream_t stream,
                              int* occupancy);

}  // namespace phi
