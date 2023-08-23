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

#include "paddle/phi/kernels/fusion/gpu/masked_multiquery_attention.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MMQAKernel(const Context& dev_ctx,
                const DenseTensor& query,
                const DenseTensor& key,
                const DenseTensor& value,
                const DenseTensor& cache_kv,
                const paddle::optional<DenseTensor>& src_mask,
                const paddle::optional<DenseTensor>& cum_offsets,
                const paddle::optional<DenseTensor>& sequence_lengths,
                const paddle::optional<DenseTensor>& rotary_tensor,
                const paddle::optional<DenseTensor>& beam_cache_offset,
                const paddle::optional<DenseTensor>& out_shift,
                const paddle::optional<DenseTensor>& out_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const float out_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                DenseTensor* out,
                DenseTensor* cache_kv_out,
                DenseTensor* beam_cache_offset_out) {
#ifndef PADDLE_WITH_HIP
  Masked_multiquery_attention_params<T> params;
  const auto& q_dims = query.dims();
  int bsz = q_dims[0];
  int num_head = q_dims[1];
  int dim_head = q_dims[2];
  int cache_bsz = cache_kv.dims()[1];
  int max_seq_len = cache_kv.dims()[3];

  int timestep = seq_len;
  float inv_sqrt_dh = 1. / sqrt(dim_head);
  bool mask_broadcast_num_heads = true;

  if (cum_offsets) {
    params.cum_offsets = cum_offsets->data<int>();
  } else {
    params.cum_offsets = nullptr;
  }

  if (src_mask) {
    if (src_mask->dims()[1] == 1) {
      mask_broadcast_num_heads = true;
    } else if (src_mask->dims()[1] == num_head) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Unknow dimension for attn_mask, the num_head(2nd) "
          "dimension is invalid, it should be 1 or num_head(%d), "
          "but got %d",
          num_head,
          src_mask->dims()[1]));
    }
    params.attn_mask = src_mask->data<T>();
    params.mask_length = src_mask->dims()[3];
  }

  if (out_scale > 0) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = cache_kv_out->data<T>();
  params.neox_rotary_style = use_neox_rotary_style;

  // params.mqa = mqa;
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
    params.beam_width = beam_cache_offset->dims()[1];
  }

  params.batch_size = bsz;
  params.cache_batch_size = cache_bsz;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_len;
  params.seq_len = seq_len;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;
  params.head_kv = cache_kv.dims()[2];
  if (out_shift) {
    DispatchFMQA<T>(dev_ctx,
                    query,
                    key,
                    value,
                    *(out_shift.get_ptr()),
                    *(out_smooth.get_ptr()),
                    params,
                    num_head,
                    dim_head,
                    out,
                    out_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  } else {
    DispatchFMQA<T>(dev_ctx,
                    query,
                    key,
                    value,
                    params,
                    num_head,
                    dim_head,
                    out,
                    out_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  }
#endif
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000
PD_REGISTER_KERNEL(masked_multiquery_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMQAKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(masked_multiquery_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMQAKernel,
                   float,
                   phi::dtype::float16) {}
#endif
