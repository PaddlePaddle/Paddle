// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "cutlass/half.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
namespace phi {
namespace sparse {
typedef void (*fp16_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const cutlass::half_t* const a,
                                         const cutlass::half_t* const b,
                                         const cutlass::half_t* const c,
                                         cutlass::half_t* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         cutlass::half_t const alpha,
                                         cutlass::half_t const beta);
typedef void (*fp32_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const float* const a,
                                         const float* const b,
                                         const float* const c,
                                         float* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         float const alpha,
                                         float const beta);
typedef void (*fp64_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const double* const a,
                                         const double* const b,
                                         const double* const c,
                                         double* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         double const alpha,
                                         double const beta);
}  // namespace sparse
}  // namespace phi
#endif
