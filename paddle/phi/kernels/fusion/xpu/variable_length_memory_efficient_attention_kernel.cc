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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const DenseTensor& seq_lens,
    const DenseTensor& kv_seq_lens,
    const paddle::optional<DenseTensor>& mask,
    const float scale,
    const bool causal,
    const int pre_cache_length,
    DenseTensor* output) {
  xpu::ctx_guard RAII_GUARD(ctx.x_context());

  using XPUType = typename XPUTypeTrait<T>::Type;

  int64_t num_batches = query.dims()[0];
  int64_t num_heads = query.dims()[1];
  int64_t kv_num_heads = key.dims()[1];
  int64_t query_seq_len = query.dims()[2];
  int64_t head_size = query.dims()[3];
  std::vector<int64_t> mask_shape = {};
  if (mask) {
    // [B, 1, S, D]
    auto mask_tensor = mask.get();
    mask_shape = common::vectorize(mask_tensor.dims());
  }

  xpu::QKVAttnParam qkv_attn_param(
      num_batches,                           /* batch */
      query_seq_len,                         /* max_seqlen */
      num_heads,                             /* head_num */
      head_size,                             /* head_dim */
      mask_shape,                            /* mask_shape */
      xpu::Activation_t::RELU,               /* act */
      -1,                                    /* last_slice_seq */
      false,                                 /* do_fc_qkv_fusion */
      -1,                                    /* hidden_dim */
      false,                                 /* is_pre_norm */
      false,                                 /* is_perchannel */
      2,                                     /* qkv_shape */
      AttnMacMaxPtrType_t::ATTN_WHOLE_BATCH, /* max_ptr_type */
      -1,                                    /* ldz */
      scale                                  /* alpha */
  );
  qkv_attn_param.key_value_head_num = kv_num_heads;

  const XPUType* mask_ptr =
      mask ? reinterpret_cast<const XPUType*>(mask.get().data<T>()) : nullptr;
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(output));
  XPUType* qk_buf = RAII_GUARD.alloc_l3_or_gm<XPUType>(
      num_batches * num_heads * query_seq_len * query_seq_len);
  float* maxptr_buf = RAII_GUARD.alloc_l3_or_gm<float>(32);
  int r = xpu::qk_attention<XPUType, XPUType, XPUType, int16_t, XPUType>(
      ctx.x_context(),                                   /* ctx */
      reinterpret_cast<const XPUType*>(query.data<T>()), /* q */
      reinterpret_cast<const XPUType*>(key.data<T>()),   /* k */
      qk_buf,                                            /* qk */
      nullptr,                                           /* max q */
      nullptr,                                           /* max k */
      maxptr_buf,                                        /* max qk */
      qkv_attn_param,                                    /* param */
      mask_ptr                                           /* mask */
  );
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::InvalidArgument("xpu::qk_attention run failed"));
  XPUType* out_tmp_buf = RAII_GUARD.alloc_l3_or_gm<XPUType>(
      num_batches * query_seq_len * num_heads * head_size);
  r = xpu::qk_v_attention<XPUType, XPUType, XPUType, int16_t>(
      ctx.x_context(),                                   /* ctx */
      qk_buf,                                            /* qk */
      reinterpret_cast<const XPUType*>(value.data<T>()), /* v */
      out_tmp_buf,                                       /* output */
      maxptr_buf,                                        /* max qk */
      nullptr,                                           /* max v */
      nullptr,                                           /* max qkv */
      qkv_attn_param                                     /* mask */
  );
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::InvalidArgument("xpu::qk_v_attention run failed"));
  r = xpu::transpose<XPUType>(
      ctx.x_context(),
      out_tmp_buf,
      out_data,
      {num_batches, query_seq_len, num_heads, head_size},
      {0, 2, 1, 3});
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::InvalidArgument("xpu::transpose run failed"));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(variable_length_memory_efficient_attention,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiHeadAttentionVariableForwardKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}
