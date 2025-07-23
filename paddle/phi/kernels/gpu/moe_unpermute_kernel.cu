// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/moe_permute_utils.h"

namespace phi {
struct __custom_bfloat164 {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
  __nv_bfloat16 z;
  __nv_bfloat16 w;
};
__device__ __nv_bfloat16 __custom_hadd(__nv_bfloat16 x, __nv_bfloat16 y) {
  return static_cast<__nv_bfloat16>(static_cast<float>(x) +
                                    static_cast<float>(y));
}
#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 64
#endif
template <bool MP>
__global__ __launch_bounds__(256) void tokens_zip_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    phi::bfloat16 *__restrict__ zipped_tokens_out,
    float *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  int local_row_fetchlist[MAX_NUM_EXPERTS];

#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
  }

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  constexpr int vecSize = 4;
  const int num_full_vec = token_length / vecSize;
  const int remaining_elems = token_length % vecSize;
  const int thread_stride = blockDim.x * vecSize;

  if constexpr (MP) {
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
      __custom_bfloat164 raw = {0.0f, 0.0f, 0.0f, 0.0f};
      int aggreg_cnt = 0;
      __custom_bfloat164 *out_ptr = reinterpret_cast<__custom_bfloat164 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        raw = *reinterpret_cast<const __custom_bfloat164 *>(
            &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset]);
        float4 token_vec = {0.0f, 0.0f, 0.0f, 0.0f};
        token_vec.x = static_cast<float>(raw.x);
        token_vec.y = static_cast<float>(raw.y);
        token_vec.z = static_cast<float>(raw.z);
        token_vec.w = static_cast<float>(raw.w);
        sum.x = __fadd_rn(token_vec.x, sum.x);
        sum.y = __fadd_rn(token_vec.y, sum.y);
        sum.z = __fadd_rn(token_vec.z, sum.z);
        sum.w = __fadd_rn(token_vec.w, sum.w);
      }
      if (aggreg_cnt > 1) {
        (*out_ptr).x = static_cast<__nv_bfloat16>(sum.x);
        (*out_ptr).y = static_cast<__nv_bfloat16>(sum.y);
        (*out_ptr).z = static_cast<__nv_bfloat16>(sum.z);
        (*out_ptr).w = static_cast<__nv_bfloat16>(sum.w);
      } else {
        *out_ptr = raw;
      }
    }

    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      float sum = 0.0f;
      __nv_bfloat16 raw = 0.0f;
      int aggreg_cnt = 0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        raw = unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        float token_val = static_cast<float>(raw);
        sum = __fadd_rn(token_val, sum);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
          (aggreg_cnt > 1) ? static_cast<__nv_bfloat16>(sum) : raw;
    }
  } else {
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      __custom_bfloat164 sum = {0.0f, 0.0f, 0.0f, 0.0f};
      __custom_bfloat164 *out_ptr = reinterpret_cast<__custom_bfloat164 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __custom_bfloat164 token_vec =
            *reinterpret_cast<const __custom_bfloat164 *>(
                &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                                 x_offset]);
        sum.x = __custom_hadd(sum.x, token_vec.x);
        sum.y = __custom_hadd(sum.y, token_vec.y);
        sum.z = __custom_hadd(sum.z, token_vec.z);
        sum.w = __custom_hadd(sum.w, token_vec.w);
      }
      *out_ptr = sum;
    }

    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      __nv_bfloat16 sum = (__nv_bfloat16)0.0f;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat16 token_val =
            unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        sum = __custom_hadd(sum, token_val);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] = sum;
    }
  }
}

template <typename T, typename Context>
void dispatch_tokens_zip(const Context &dev_ctx,
                         const DenseTensor &unzipped_tokens,
                         const DenseTensor &zipped_expertwise_rowmap,
                         const DenseTensor &expert_routemap_topk,
                         const DenseTensor &unzipped_token_probs,
                         DenseTensor *zipped_tokens,
                         DenseTensor *zipped_probs_topk,
                         const int total_zipped_tokens_num,
                         const int num_experts,
                         const int token_length,
                         const int topk,
                         const bool MP) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;

  // Map data types to C++ types
  if (unzipped_token_probs.dtype() == paddle::DataType::FLOAT32) {
    if (MP == true) {
      tokens_zip_kernel<true><<<grid, block, 0, dev_ctx.stream()>>>(
          unzipped_tokens.data<phi::bfloat16>(),
          zipped_expertwise_rowmap.data<int>(),
          expert_routemap_topk.data<int>(),
          unzipped_token_probs.data<float>(),
          zipped_tokens->data<phi::bfloat16>(),
          zipped_probs_topk->data<float>(),
          total_zipped_tokens_num,
          token_length,
          num_experts,
          topk);
    } else {
      tokens_zip_kernel<false><<<grid, block, 0, dev_ctx.stream()>>>(
          unzipped_tokens.data<phi::bfloat16>(),
          zipped_expertwise_rowmap.data<int>(),
          expert_routemap_topk.data<int>(),
          unzipped_token_probs.data<float>(),
          zipped_tokens->data<phi::bfloat16>(),
          zipped_probs_topk->data<float>(),
          total_zipped_tokens_num,
          token_length,
          num_experts,
          topk);
    }
  }
}

template <typename T, typename Context>
void MoeUnpermuteKernel(const Context &dev_ctx,
                        const DenseTensor &unzipped_tokens,
                        const DenseTensor &zipped_expertwise_rowmap,
                        const DenseTensor &expert_routemap_topk,
                        const DenseTensor &unzipped_token_probs,
                        const int total_zipped_tokens_num,
                        const int num_experts,
                        const bool MP,
                        DenseTensor *zipped_tokens,
                        DenseTensor *zipped_probs_topk) {
  const int rows = unzipped_tokens.dims()[0];
  const int cols = unzipped_tokens.dims()[1];
  PADDLE_ENFORCE_LE(
      num_experts,
      MAX_NUM_EXPERTS,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          MAX_NUM_EXPERTS,
          num_experts));
  const int topk = expert_routemap_topk.dims()[1];
  dev_ctx.template Alloc<T>(zipped_tokens);
  dev_ctx.template Alloc<float>(zipped_probs_topk);
  if (unzipped_tokens.numel() == 0) return;  // 0-size tensor
  void *zipped_probs_topk_ptr =
      reinterpret_cast<void *>(zipped_probs_topk->data<float>());
  cudaMemsetAsync(zipped_probs_topk_ptr,
                  0,
                  sizeof(float) * total_zipped_tokens_num * topk,
                  dev_ctx.stream());

  dispatch_tokens_zip<T, Context>(dev_ctx,
                                  unzipped_tokens,
                                  zipped_expertwise_rowmap,
                                  expert_routemap_topk,
                                  unzipped_token_probs,
                                  zipped_tokens,
                                  zipped_probs_topk,
                                  total_zipped_tokens_num,
                                  num_experts,
                                  cols,
                                  topk,
                                  MP);
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_unpermute,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeUnpermuteKernel,
                   phi::dtype::bfloat16) {}
