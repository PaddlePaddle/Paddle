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

#include "paddle/phi/kernels/fusion/gpu/masked_multihead_attention.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MMHAKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const paddle::optional<DenseTensor>& bias,
                const paddle::optional<DenseTensor>& src_mask,
                const paddle::optional<DenseTensor>& sequence_lengths,
                const paddle::optional<DenseTensor>& rotary_tensor,
                const paddle::optional<DenseTensor>& beam_cache_offset,
                const DenseTensor& cache_kv,
                const paddle::optional<DenseTensor>& qkv_out_scale,
                const paddle::optional<DenseTensor>& out_linear_shift,
                const paddle::optional<DenseTensor>& out_linear_smooth,
                int beam_size,
                int rotary_emb_dims,
                const bool mask_broadcast_num_heads,
                const bool compute_bias,
                const bool use_neox_rotary_style,
                const float out_linear_in_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                DenseTensor* out,
                DenseTensor* cache_kv_out,
                DenseTensor* beam_cache_offset_out) {
  const auto& x_dims = x.dims();
  int bsz = x_dims[0];
  int num_head = x_dims[2];
  int dim_head = x_dims[3];
  int timestep = src_mask->dims()[3] - 1;
  int cache_bsz = cache_kv.dims()[1];
  int max_seq_len = cache_kv.dims()[3];
  float inv_sqrt_dh = 1. / sqrt(dim_head);

  if (out_linear_in_scale > 0) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  Masked_multihead_attention_params<T> params;
  params.attn_mask = src_mask->data<T>();
  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = const_cast<T*>(cache_kv_out->data<T>());
  params.neox_rotary_style = use_neox_rotary_style;
  params.mask_length = src_mask->dims()[3];

  if (sequence_lengths) {
    params.sequence_lengths = sequence_lengths->data<int>();
  }
  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
  } else {
    params.rotary_emb = nullptr;
  }

  if (beam_cache_offset) {
    params.beam_cache_offset = beam_cache_offset->data<int>();
  }

  params.add_qkv_bias = compute_bias;
  if (compute_bias) {
    // Because we may not add qkv_bias, so here we cast to T*.
    // Author(zhengzekang).
    params.qkv_bias = const_cast<T*>(bias->data<T>());
  }

  params.batch_size = bsz;
  params.cache_batch_size = cache_bsz;
  params.beam_width = beam_size;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_len;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;
  if (out_linear_shift) {
    DispatchFMHA<T>(dev_ctx,
                    x,
                    *(out_linear_shift.get_ptr()),
                    *(out_linear_smooth.get_ptr()),
                    params,
                    num_head,
                    dim_head,
                    out,
                    qkv_out_scale.get_ptr(),
                    out_linear_in_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  } else {
    DispatchFMHA<T>(dev_ctx,
                    x,
                    params,
                    num_head,
                    dim_head,
                    out,
                    qkv_out_scale.get_ptr(),
                    out_linear_in_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(masked_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMHAKernel,
                   float,
#if defined(__CUDACC__) && CUDA_VERSION >= 11000
                   phi::dtype::bfloat16,
#endif
                   phi::dtype::float16) {
}
