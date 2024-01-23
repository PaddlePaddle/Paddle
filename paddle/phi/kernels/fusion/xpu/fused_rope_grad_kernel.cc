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

#include "paddle/phi/kernels/fusion/xpu/fused_rope_utils.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedRopeGradKernel(const Context& dev_ctx,
                         const paddle::optional<DenseTensor>& sin,
                         const paddle::optional<DenseTensor>& cos,
                         const paddle::optional<DenseTensor>& position_ids,
                         const DenseTensor& dout_q,
                         const paddle::optional<DenseTensor>& dout_k,
                         const paddle::optional<DenseTensor>& dout_v,
                         bool use_neox_rotary_style,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  using XPUT = typename XPUTypeTrait<T>::Type;
  if (dout_q.numel() <= 0) {
    return;
  }

  int64_t batch_size = dout_q.dims()[0];
  int64_t seq_len = dout_q.dims()[1];
  int64_t num_heads = dout_q.dims()[2];
  int64_t head_dim = dout_q.dims()[3];
  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t sin_cos_len = batch_size * seq_len * head_dim;
  auto* sin_data = RAII_GUARD.alloc_l3_or_gm<XPUT>(sin_cos_len);
  auto* cos_data = RAII_GUARD.alloc_l3_or_gm<XPUT>(sin_cos_len);

  if (sin.get_ptr() && cos.get_ptr()) {
    PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                      cos.get_ptr()->dims(),
                      phi::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "recieved sin's dims is {%s}, cos's dims is {%s}.",
                          sin.get_ptr()->dims(),
                          cos.get_ptr()->dims()));
  }

  XPUGetSinCosData<XPUT, Context>(
      dev_ctx, sin, position_ids, sin_data, batch_size, seq_len, head_dim);
  XPUGetSinCosData<XPUT, Context>(
      dev_ctx, cos, position_ids, cos_data, batch_size, seq_len, head_dim);

  if (use_neox_rotary_style) {
    // TODO(lijin23): support rotary_embedding every_two.
    PADDLE_THROW(
        phi::errors::Unimplemented("XPU do not support rotary_embedding_grad "
                                   "with use_neox_rotary_style set."));
  } else {
    auto* dq_data = reinterpret_cast<XPUT*>(dev_ctx.template Alloc<T>(dq));
    XPUFusedRotaryHalf<XPUT, Context>(
        dev_ctx,
        reinterpret_cast<const XPUT*>(dout_q.data<T>()),
        sin_data,
        cos_data,
        dq_data,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        true);

    if (dout_k.get_ptr()) {
      auto* dk_data = reinterpret_cast<XPUT*>(dev_ctx.template Alloc<T>(dk));
      XPUFusedRotaryHalf<XPUT, Context>(
          dev_ctx,
          reinterpret_cast<const XPUT*>(dout_k->data<T>()),
          sin_data,
          cos_data,
          dk_data,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          true);
    }

    if (dout_v.get_ptr()) {
      auto* dv_data = reinterpret_cast<XPUT*>(dev_ctx.template Alloc<T>(dv));
      XPUFusedRotaryHalf<XPUT, Context>(
          dev_ctx,
          reinterpret_cast<const XPUT*>(dout_v->data<T>()),
          sin_data,
          cos_data,
          dv_data,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          true);
    }
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
