// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <type_traits>
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/sparse/gpu/cutlass_generator/all_gemm_operations.h"

namespace phi {
namespace sparse {

// To reduce tuning time, map shape (m,n,k) to (m/features_num_range,n,k) so
// that shapes within this range share the same key.
constexpr int features_num_range = 10000;

template <int ComputeCapability,
          bool TransposeA,
          bool TransposeB,
          typename Input,
          typename Output,
          typename IntT>
void GatherGemmScatterDriver(
    const phi::GPUContext& ctx,
    const size_t key,
    const Input* const a,
    const Input* const b,
    const Output* const c,
    Output* const d,
    const int& m,
    const int& n,
    const int& k,
    const IntT* a_indices,
    const IntT* b_indices,
    const IntT* c_d_indices,
    Output alpha,
    Output beta,
    cutlass::device_memory::allocation<uint8_t>* const workspace_ptr) {
  PADDLE_THROW(common::errors::Unimplemented(
      "gather_gemm_scatter fusion only supports "
      "fp16_nn, fp32_nn, fp32_nt and fp32_tn now."));
}

#define EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(                       \
    compute_capability, transpose_a, transpose_b, in_type, out_type, kernels) \
  template <>                                                                 \
  inline void GatherGemmScatterDriver<compute_capability,                     \
                                      transpose_a,                            \
                                      transpose_b,                            \
                                      in_type,                                \
                                      out_type,                               \
                                      int32_t>(                               \
      const phi::GPUContext& ctx,                                             \
      const size_t key,                                                       \
      const in_type* const a,                                                 \
      const in_type* const b,                                                 \
      const out_type* const c,                                                \
      out_type* const d,                                                      \
      const int& m,                                                           \
      const int& n,                                                           \
      const int& k,                                                           \
      const int32_t* a_indices,                                               \
      const int32_t* b_indices,                                               \
      const int32_t* c_d_indices,                                             \
      out_type alpha,                                                         \
      out_type beta,                                                          \
      cutlass::device_memory::allocation<uint8_t>* const workspace_ptr) {     \
    auto* tuner =                                                             \
        autotune::MakeGatherGemmScatterTuner<transpose_a, transpose_b>(       \
            kernels[0]);                                                      \
    for (auto i = 1; i < kernels.size(); i++) tuner->AddCallBack(kernels[i]); \
    tuner->Run(ctx,                                                           \
               key,                                                           \
               alpha,                                                         \
               beta,                                                          \
               ctx,                                                           \
               a,                                                             \
               b,                                                             \
               c,                                                             \
               d,                                                             \
               m,                                                             \
               n,                                                             \
               k,                                                             \
               a_indices,                                                     \
               b_indices,                                                     \
               c_d_indices,                                                   \
               workspace_ptr);                                                \
  }

EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(75,
                                               false,
                                               false,
                                               phi::dtype::float16,
                                               phi::dtype::float16,
                                               sm75_fp16_nn_kernels)
EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(
    75, false, false, phi::dtype::float16, float, sm75_fp32_nn_kernels)
EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(80,
                                               false,
                                               false,
                                               phi::dtype::float16,
                                               phi::dtype::float16,
                                               sm80_fp16_nn_kernels)
EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(
    80, false, false, float, float, sm80_fp32_nn_kernels)
EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(
    80, false, true, float, float, sm80_fp32_nt_kernels)
EXPLICIT_SPECIALIZE_GATHER_GEMM_SCATTER_DRIVER(
    80, true, false, float, float, sm80_fp32_tn_kernels)

}  // namespace sparse
}  // namespace phi
#endif
