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

template <typename T>
typename std::enable_if<std::is_same<T, phi::dtype::float16>::value, void>::type
GatherGemmScatter(const phi::GPUContext& ctx,
                  const T* const a,
                  const T* const b,
                  const T* const c,
                  T* const d,
                  const int& m,
                  const int& n,
                  const int& k,
                  const int32_t* a_indices,
                  const int32_t* b_indices,
                  const int32_t* c_d_indices,
                  T alpha,
                  T beta) {
  auto* tuner = autotune::MakeGatherGemmScatterTuner<T>(fp16_kernels[0]);
  for (auto i = 1; i < fp16_kernels.size(); i++)
    tuner->AddCallBack(fp16_kernels[i]);

  size_t key = autotune::GenKey(m / features_num_range, n, k);

  tuner->Run(ctx,
             key,
             a,
             b,
             c,
             d,
             m,
             n,
             k,
             a_indices,
             b_indices,
             c_d_indices,
             alpha,
             beta);
}

template <typename T>
typename std::enable_if<std::is_same<T, float>::value, void>::type
GatherGemmScatter(const phi::GPUContext& ctx,
                  const T* const a,
                  const T* const b,
                  const T* const c,
                  T* const d,
                  const int& m,
                  const int& n,
                  const int& k,
                  const int32_t* a_indices,
                  const int32_t* b_indices,
                  const int32_t* c_d_indices,
                  T alpha,
                  T beta) {
  auto* tuner = autotune::MakeGatherGemmScatterTuner<T>(fp32_kernels[0]);
  for (auto i = 1; i < fp32_kernels.size(); i++)
    tuner->AddCallBack(fp32_kernels[i]);

  size_t key = autotune::GenKey(m / features_num_range, n, k);

  tuner->Run(ctx,
             key,
             a,
             b,
             c,
             d,
             m,
             n,
             k,
             a_indices,
             b_indices,
             c_d_indices,
             alpha,
             beta);
}

template <typename T>
static void dispatchKernel(const GPUContext& dev_ctx,
                           const void* const a,
                           const void* const b,
                           const void* const c,
                           void* const d,
                           const int m,
                           const int n,
                           const int k,
                           const void* a_indices,
                           const void* c_d_indices,
                           const T alpha,
                           const T beta,
                           const phi::DataType type) {
  if (type == phi::DataType::FLOAT16) {
    GatherGemmScatter(dev_ctx,
                      static_cast<const phi::dtype::float16*>(a),
                      static_cast<const phi::dtype::float16*>(b),
                      static_cast<const phi::dtype::float16*>(c),
                      static_cast<phi::dtype::float16*>(d),
                      m,
                      n,
                      k,
                      static_cast<const int32_t*>(a_indices),
                      nullptr,
                      static_cast<const int32_t*>(c_d_indices),
                      static_cast<phi::dtype::float16>(alpha),
                      static_cast<phi::dtype::float16>(beta));
  } else if (type == phi::DataType::FLOAT32) {
    GatherGemmScatter(dev_ctx,
                      static_cast<const float*>(a),
                      static_cast<const float*>(b),
                      static_cast<const float*>(c),
                      static_cast<float*>(d),
                      m,
                      n,
                      k,
                      static_cast<const int32_t*>(a_indices),
                      nullptr,
                      static_cast<const int32_t*>(c_d_indices),
                      static_cast<float>(alpha),
                      static_cast<float>(beta));
  }
}

}  // namespace sparse
}  // namespace phi
#endif
