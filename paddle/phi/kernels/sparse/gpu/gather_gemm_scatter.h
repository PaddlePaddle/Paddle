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
#ifdef PADDLE_WITH_CUTLASS
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/sparse/gpu/cutlass_generator/build/generated/gemm/all_gemm_operations.h"

namespace phi {
namespace sparse {

// To reduce tuning time, map shape (m,n,k) to (m/features_num_range,n,k) so
// that shapes in this range share the same key.
constexpr int features_num_range = 10000;

#define DEFINE_GATHER_GEMM_SCATTER_DRIVER(dtype, kernels)                     \
  template <typename T, typename IntT>                                        \
  typename std::enable_if<std::is_same<T, dtype>::value &&                    \
                              std::is_same<IntT, int32_t>::value,             \
                          void>::type                                         \
  GatherGemmScatterDriver(const phi::GPUContext& ctx,                         \
                          const T* const a,                                   \
                          const T* const b,                                   \
                          const T* const c,                                   \
                          T* const d,                                         \
                          const int& m,                                       \
                          const int& n,                                       \
                          const int& k,                                       \
                          const IntT* a_indices,                              \
                          const IntT* c_d_indices,                            \
                          T alpha,                                            \
                          T beta) {                                           \
    auto* tuner = autotune::MakeGatherGemmScatterTuner(kernels[0]);           \
    for (auto i = 1; i < kernels.size(); i++) tuner->AddCallBack(kernels[i]); \
    size_t key = autotune::GenKey(m / features_num_range, n, k);              \
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
               c_d_indices);                                                  \
  }

template <typename T, typename IntT>
typename std::enable_if<std::is_same<T, double>::value ||
                            !std::is_same<IntT, int32_t>::value,
                        void>::type
GatherGemmScatterDriver(const phi::GPUContext& ctx,
                        const T* const a,
                        const T* const b,
                        const T* const c,
                        T* const d,
                        const int& m,
                        const int& n,
                        const int& k,
                        const IntT* a_indices,
                        const IntT* c_d_indices,
                        T alpha,
                        T beta) {}

DEFINE_GATHER_GEMM_SCATTER_DRIVER(phi::dtype::float16, fp16_nn_kernels)
DEFINE_GATHER_GEMM_SCATTER_DRIVER(float, fp32_nn_kernels)

}  // namespace sparse
}  // namespace phi
#endif
