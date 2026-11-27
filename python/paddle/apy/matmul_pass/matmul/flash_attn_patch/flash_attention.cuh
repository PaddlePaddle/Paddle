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

// FlashAttention based backend for `memory_efficient_attention + epilogue`.
//
// The FlashAttention forward kernel comes from `AP_FLASHATTN_DIR` and is
// overlaid by the patched copies in this directory (`flash.h`,
// `flash_fwd_kernel.h`) so that its epilogue applies the AP generated
// `VariadicFunctor` in registers. This header only builds the
// `Flash_fwd_params` and dispatches to the right head-dim kernel; the generated
// subgraph .so compiles the FA kernel template itself.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "cutlass_patch/batched_matrix_coord.h"
#include "params.h"  // NOLINT

namespace ap {

using bfloat16 = nv_bfloat16;
using MatrixCoord = cutlass_patch::BatchedMatrixCoord;

// Convert CUDA data type to cutlass data type (the FA kernel expects the
// cutlass scalar types).
template <typename T>
struct CutlassDataType {
  using Type = T;
};

template <>
struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

// Small device workspace for the log-sum-exp buffer. `libflashattn`'s helper
// lives in cutlass/util (not shipped by the FA fork headers), so keep a local
// one for the standalone generated .so.
static void *FlashAttnWorkspace(size_t size) {
  static void *ptr = nullptr;
  static size_t capacity = 0;
  if (capacity < size) {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
    cudaMalloc(&ptr, size);
    capacity = size;
  }
  return ptr;
}

}  // namespace ap

// FlashAttention forward headers, provided by AP_FLASHATTN_DIR.
#include "cuda_utils.h"  // NOLINT
#include "flash.h"       // NOLINT

#if !FLASH_ATTN_WITH_TORCH
// The FA forward headers reference this helper (defined in the FA library's
// cuda_utils.cu). The generated subgraph .so is standalone, so provide it here.
namespace at {
namespace cuda {
inline cudaDeviceProp *getCurrentDeviceProperties(int id) {
  static cudaDeviceProp prop;
  int dev = id;
  if (dev < 0) {
    cudaGetDevice(&dev);
  }
  cudaGetDeviceProperties(&prop, dev);
  return &prop;
}
}  // namespace cuda
}  // namespace at
#endif

#include "flash_fwd_launch_template.h"  // NOLINT

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

constexpr int RoundUpToMultiple(int x, int m) { return (x + m - 1) / m * m; }

// FlashAttention dispatches on a head-dim bucket (`HEADDIM_SWITCH`); `params.d`
// keeps the actual head dim so non-bucket sizes (still multiple of 8) work.
template <typename scalar_t,
          bool IsCausal,
          int HeadDim,
          typename VariadicFunctorArgs>
inline void LaunchFlashAttentionFwd(Flash_fwd_params *p,
                                    cudaStream_t stream,
                                    const VariadicFunctorArgs &variadic_args) {
  // The functor Arguments travel inside Flash_fwd_params (a POD buffer) so the
  // FA kernel signature stays `flash_fwd_kernel(Flash_fwd_params)`.
  static_assert(sizeof(VariadicFunctorArgs) <= sizeof(p->variadic_args),
                "AP epilogue Arguments do not fit in Flash_fwd_params");
  std::memcpy(p->variadic_args, &variadic_args, sizeof(VariadicFunctorArgs));
  if constexpr (HeadDim <= 32) {
    run_mha_fwd_hdim32<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 64) {
    run_mha_fwd_hdim64<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 96) {
    run_mha_fwd_hdim96<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 128) {
    run_mha_fwd_hdim128<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 160) {
    run_mha_fwd_hdim160<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 192) {
    run_mha_fwd_hdim192<scalar_t, IsCausal, false, false>(*p, stream);
  } else if constexpr (HeadDim <= 224) {
    run_mha_fwd_hdim224<scalar_t, IsCausal, false, false>(*p, stream);
  } else {
    run_mha_fwd_hdim256<scalar_t, IsCausal, false, false>(*p, stream);
  }
}

}  // namespace detail

// Fused multi-head attention (fprop) + elementwise epilogue, single pass.
//   O = VariadicFunctor(softmax(Q*K^T*scale) * V, variadic_args, coord)
//
// Same contract as the cutlass example-41 backend: BMHK layout, no GQA, no
// dropout / attn_mask, optional causal. Coordinate semantics:
//   args.b / args.h = batch / head, coord.row = m, coord.column = dv.
template <typename ElementT,
          typename ElementComputeT,
          template <typename>
          class VariadicFunctor,
          int HeadDim,
          bool IsCausal = false,
          int ConfigId = 0>
void AttentionVariadicFusion(
    const AttentionEpilogueParams &params,
    const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
  using scalar_t = typename CutlassDataType<ElementT>::Type;

  cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(params.stream_ptr);

  Flash_fwd_params p;
  std::memset(&p, 0, sizeof(p));

  p.is_bf16 = std::is_same<ElementT, bfloat16>::value;

  p.q_ptr = const_cast<void *>(params.query);
  p.k_ptr = const_cast<void *>(params.key);
  p.v_ptr = const_cast<void *>(params.value);
  p.o_ptr = params.output;

  p.q_row_stride = static_cast<uint32_t>(params.q_strideM);
  p.k_row_stride = static_cast<uint32_t>(params.k_strideM);
  p.v_row_stride = static_cast<uint32_t>(params.v_strideM);
  p.q_head_stride = static_cast<uint32_t>(params.q_strideH);
  p.k_head_stride = static_cast<uint32_t>(params.k_strideH);
  p.v_head_stride = static_cast<uint32_t>(params.v_strideH);
  p.o_row_stride = static_cast<uint32_t>(params.o_strideM);
  p.o_head_stride = static_cast<uint32_t>(params.o_strideH);

  p.q_batch_stride = static_cast<uint32_t>(params.q_strideB);
  p.k_batch_stride = static_cast<uint32_t>(params.k_strideB);
  p.v_batch_stride = static_cast<uint32_t>(params.v_strideB);
  p.o_batch_stride = static_cast<uint32_t>(params.o_strideB);

  p.cu_seqlens_q = nullptr;
  p.cu_seqlens_k = nullptr;
  p.varlen_padded_input = false;

  p.p_ptr = nullptr;

  // The FA forward epilogue always writes the log-sum-exp, and the first
  // threadblock always writes the (seed, offset) pair to `rng_state`, even in
  // inference / no-dropout mode. Back both with the workspace.
  const size_t lse_bytes =
      static_cast<size_t>(params.B) * params.H * params.M * sizeof(float);
  const size_t lse_offset = (lse_bytes + 7u) & ~static_cast<size_t>(7u);
  unsigned char *workspace = reinterpret_cast<unsigned char *>(
      FlashAttnWorkspace(lse_offset + 2 * sizeof(uint64_t)));
  p.softmax_lse_ptr = workspace;
  p.rng_state = reinterpret_cast<uint64_t *>(workspace + lse_offset);

  p.b = params.B;
  p.h = params.H;
  p.h_k = params.H;
  p.h_h_k_ratio = 1;
  p.seqlen_q = params.M;
  p.seqlen_k = params.N;
  p.seqlen_q_rounded = detail::RoundUpToMultiple(params.M, 128);
  p.seqlen_k_rounded = detail::RoundUpToMultiple(params.N, 128);
  p.d = params.Dh;
  p.d_rounded = detail::RoundUpToMultiple(params.Dh, 32);

  const float scale = params.scale > 0.f
                          ? params.scale
                          : (1.0f / std::sqrt(static_cast<float>(params.Dh)));
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * M_LOG2E;
  p.unscale_softmax = 1.f;

  // No dropout.
  p.p_dropout = 1.f;
  p.p_dropout_in_uint8_t = 0;
  p.rp_dropout = 1.f;
  p.scale_softmax_rp_dropout = scale;

  p.is_causal = IsCausal;
  p.unpadded_lse = false;
  p.total_q = 0;

  detail::LaunchFlashAttentionFwd<scalar_t, IsCausal, HeadDim>(
      &p, stream, variadic_args);

#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
}

}  // namespace ap
