// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <cstdint>
#include <iostream>

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"
#include "cutlass_patch/cuda/cutlass_common.cuh"
#include "cutlass_patch/cuda/default_config_id.h"
#include "cutlass_patch/fmha/kernel_forward.h"

#include "params.h"  // NOLINT

#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                    \
  {                                                                         \
    cudaError_t err = (func);                                               \
    if (err != cudaSuccess) {                                               \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] CUDA error("    \
                << err << "), " << cudaGetErrorString(err) << " when call " \
                << #func << std::endl;                                      \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }
#endif

namespace ap {

namespace detail {

// Threadblock/key tile selection by head dim. kMaxK must be >= head_dim and
// head_dim_value; the FMHA kernel single-pass path kicks in when
// kMaxK <= kKeysPerBlock.
template <int HeadDim>
struct FmhaConfig {
  static constexpr int kQueriesPerBlock = (HeadDim <= 64) ? 64 : 32;
  static constexpr int kKeysPerBlock = (HeadDim <= 64) ? 64 : 128;
  static constexpr int kMaxK = (HeadDim <= 64) ? 64 : 128;
};

}  // namespace detail

// Fused multi-head attention (fprop) + elementwise epilogue, single pass.
//   O = VariadicFunctor(softmax(Q*K^T*scale) * V, variadic_args, coord)
//
// The variadic functor is applied inside the FMHA output op
// (MemoryEfficientAttentionNormalizeWithVariadic) right after softmax
// normalization, so there is no extra global read/write.
//
// Coordinate semantics seen by the functor (BMHK output [B, M, H, Dv]):
//   args.b / args.h = batch / head, coord.row = m (query), coord.column = dv.
template <typename ElementT,
          typename ElementComputeT,
          template <typename>
          class VariadicFunctor,
          int HeadDim,
          bool IsCausal = false,
          int ConfigId = DefaultConfig::kConfigId>
void AttentionVariadicFusion(
    const AttentionEpilogueParams &params,
    const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
  using scalar_t = typename CutlassDataType<ElementT>::Type;
  using Cfg = detail::FmhaConfig<HeadDim>;
  using Attention = AttentionKernel<scalar_t,
                                    cutlass::arch::Sm80,
                                    /*isAligned=*/true,
                                    Cfg::kQueriesPerBlock,
                                    Cfg::kKeysPerBlock,
                                    Cfg::kMaxK,
                                    /*kSupportsDropout=*/false,
                                    /*kSupportsBias=*/false,
                                    DefaultToBatchHook,
                                    VariadicFunctor>;

  cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(params.stream_ptr);

  typename Attention::Params p;
  p.query_ptr = reinterpret_cast<scalar_t *>(const_cast<void *>(params.query));
  p.key_ptr = reinterpret_cast<scalar_t *>(const_cast<void *>(params.key));
  p.value_ptr = reinterpret_cast<scalar_t *>(const_cast<void *>(params.value));
  p.output_ptr = reinterpret_cast<scalar_t *>(params.output);
  p.logsumexp_ptr = nullptr;
  p.output_accum_ptr = nullptr;
  // Epilogue operands; b/h are filled per threadblock inside the output op.
  p.variadic_args = variadic_args;
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    const size_t bytes = static_cast<size_t>(params.B) * params.M * params.H *
                         params.Dv * sizeof(typename Attention::output_accum_t);
    p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t *>(
        GetWorkspace(bytes));
  }

  p.scale = params.scale > 0.f
                ? params.scale
                : (1.0f / std::sqrt(static_cast<float>(params.Dh)));
  p.num_heads = params.H;
  p.num_batches = params.B;
  p.head_dim = params.Dh;
  p.head_dim_value = params.Dv;
  p.num_queries = params.M;
  p.num_keys = params.N;
  p.num_keys_absolute = params.N;
  if (IsCausal) {
    p.custom_mask_type = Attention::CausalFromTopLeft;
  }

  // All tensors are BMHK.
  p.q_strideH = static_cast<int32_t>(params.q_strideH);
  p.k_strideH = static_cast<int32_t>(params.k_strideH);
  p.v_strideH = static_cast<int32_t>(params.v_strideH);
  p.q_strideM = static_cast<int32_t>(params.q_strideM);
  p.k_strideM = static_cast<int32_t>(params.k_strideM);
  p.v_strideM = static_cast<int32_t>(params.v_strideM);
  p.q_strideB = params.q_strideB;
  p.k_strideB = params.k_strideB;
  p.v_strideB = params.v_strideB;
  p.o_strideM = static_cast<int32_t>(params.o_strideM);

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = static_cast<int>(sizeof(typename Attention::SharedStorage));
  if (smem_bytes > 0xc000) {
    CHECK_CUDA(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  if (!Attention::check_supported(p)) {
    std::cerr << "[AttentionVariadicFusion] kernel does not support these "
                 "inputs"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
}

}  // namespace ap
