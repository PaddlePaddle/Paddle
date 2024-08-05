// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/fusion/xpu/fused_rope_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const paddle::optional<DenseTensor>& k,
                     const paddle::optional<DenseTensor>& v,
                     const paddle::optional<DenseTensor>& sin,
                     const paddle::optional<DenseTensor>& cos,
                     const paddle::optional<DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     float rotary_emb_base,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (q.numel() <= 0) {
    return;
  }
  PADDLE_ENFORCE_EQ(
      time_major,
      false,
      common::errors::InvalidArgument("time_major is not supported in xpu"));
  int64_t batch_size = q.dims()[0];
  int64_t seq_len = q.dims()[1];
  int64_t num_heads = q.dims()[2];
  int64_t head_dim = q.dims()[3];
  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  int64_t sin_cos_len = batch_size * seq_len * head_dim;
  if (use_neox_rotary_style) {
    auto* sin_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(sin_cos_len);
    auto* cos_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(sin_cos_len);
    if (sin.get_ptr() && cos.get_ptr()) {
      PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                        cos.get_ptr()->dims(),
                        common::errors::InvalidArgument(
                            "The dims of sin and cos must be the same. But "
                            "received sin's dims is {%s}, cos's dims is {%s}.",
                            sin.get_ptr()->dims(),
                            cos.get_ptr()->dims()));
    }
    XPUGetSinCosData<XPUType, Context>(
        dev_ctx, sin, position_ids, sin_data, batch_size, seq_len, head_dim);
    XPUGetSinCosData<XPUType, Context>(
        dev_ctx, cos, position_ids, cos_data, batch_size, seq_len, head_dim);
    if (!k) {
      auto* outq_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_q));
      int ret = xpu::rotary_embedding_v3_single<XPUType, XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(q.data<T>()),
          cos_data,
          sin_data,
          outq_data,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          {seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rotary_embedding_v3_single");
    } else {
      int64_t num_heads_k = k->dims()[2];
      auto* outq_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_q));
      auto* outk_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_k));
      int ret = xpu::rotary_embedding_v3<XPUType, XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(q.data<T>()),
          reinterpret_cast<const XPUType*>(k->data<T>()),
          cos_data,
          sin_data,
          outq_data,
          outk_data,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          {seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, 1},
          {seq_len * num_heads_k * head_dim,
           num_heads_k * head_dim,
           head_dim,
           1},
          num_heads_k);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rotary_embedding_v3");
    }

    if (v) {
      int64_t num_heads_v = v->dims()[2];
      auto* outv_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_v));
      int ret = xpu::rotary_embedding_v3_single<XPUType, XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(v->data<T>()),
          cos_data,
          sin_data,
          outv_data,
          batch_size,
          seq_len,
          num_heads_v,
          head_dim,
          {seq_len * num_heads_v * head_dim,
           num_heads_v * head_dim,
           head_dim,
           1});
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rotary_embedding_v3_single");
    }
  } else {
    auto* sin_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(sin_cos_len);
    auto* cos_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(sin_cos_len);
    if (sin.get_ptr() && cos.get_ptr()) {
      PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                        cos.get_ptr()->dims(),
                        common::errors::InvalidArgument(
                            "The dims of sin and cos must be the same. But "
                            "received sin's dims is {%s}, cos's dims is {%s}.",
                            sin.get_ptr()->dims(),
                            cos.get_ptr()->dims()));
    }
    XPUGetSinCosData<XPUType, Context>(
        dev_ctx, sin, position_ids, sin_data, batch_size, seq_len, head_dim);
    XPUGetSinCosData<XPUType, Context>(
        dev_ctx, cos, position_ids, cos_data, batch_size, seq_len, head_dim);
    if (head_dim * sizeof(T) <= 1024 && head_dim % 64 == 0 && k) {
      int64_t num_heads_k = k->dims()[2];
      auto* outq_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_q));
      auto* outk_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_k));
      int ret = xpu::rotary_no_freqs_qk_embedding_v2<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(q.data<T>()),
          reinterpret_cast<const XPUType*>(k->data<T>()),
          sin_data,
          cos_data,
          outq_data,
          outk_data,
          {batch_size, seq_len, num_heads, head_dim},
          {batch_size, seq_len, 1, head_dim},
          {seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, 1},
          {seq_len * head_dim, head_dim, head_dim, 1},
          num_heads_k);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rotary_no_freqs_qk_embedding_v2");
    } else {
      auto* outq_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_q));
      XPUFusedRotaryHalf<XPUType, Context>(
          dev_ctx,
          reinterpret_cast<const XPUType*>(q.data<T>()),
          sin_data,
          cos_data,
          outq_data,
          batch_size,
          seq_len,
          num_heads,
          head_dim);

      if (k) {
        int64_t num_heads_k = k->dims()[2];
        auto* outk_data =
            reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_k));
        XPUFusedRotaryHalf<XPUType, Context>(
            dev_ctx,
            reinterpret_cast<const XPUType*>(k->data<T>()),
            sin_data,
            cos_data,
            outk_data,
            batch_size,
            seq_len,
            num_heads_k,
            head_dim);
      }
    }

    if (v) {
      int64_t num_heads_v = v->dims()[2];
      auto* outv_data =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out_v));
      XPUFusedRotaryHalf<XPUType, Context>(
          dev_ctx,
          reinterpret_cast<const XPUType*>(v->data<T>()),
          sin_data,
          cos_data,
          outv_data,
          batch_size,
          seq_len,
          num_heads_v,
          head_dim);
    }
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
