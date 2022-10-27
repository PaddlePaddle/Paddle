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

#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/device_memory.h"
#include "examples/common/helper.h"
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
fp16_gather_gemm_scatter getBestFp16Kernel(const int M,
                                           const int K,
                                           const int N);
fp32_gather_gemm_scatter getBestFp32Kernel(const int M,
                                           const int K,
                                           const int N);
fp64_gather_gemm_scatter getBestFp64Kernel(const int M,
                                           const int K,
                                           const int N);
void cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_h1688gemm_64x128_32x2_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_h1688gemm_128x64_32x2_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688f16gemm_256x64_16x4_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688f16gemm_64x128_16x6_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4(
    const GPUContext& dev_ctx,
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
void cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4(const GPUContext& dev_ctx,
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
void cutlass_tensorop_d884gemm_16x32_16x5_nn_align1(const GPUContext& dev_ctx,
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
void cutlass_tensorop_d884gemm_32x16_16x5_nn_align1(const GPUContext& dev_ctx,
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
