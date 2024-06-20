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
#include "paddle/phi/backends/xpu/enforce_xpu.h"

namespace phi {
namespace fusion {
template <typename XPUType, typename Context>
void XPUGetSinCosData(const Context& dev_ctx,
                      const paddle::optional<DenseTensor>& sin_cos,
                      const paddle::optional<DenseTensor>& position_ids,
                      XPUType* sin_cos_data,
                      int64_t batch_size,
                      int64_t seq_len,
                      int64_t head_dim) {
  if (sin_cos.get_ptr()) {
    auto sin_cos_dims = sin_cos.get_ptr()->dims();
    int64_t dims_size = sin_cos_dims.size();
    PADDLE_ENFORCE_EQ(
        (dims_size == 2 || dims_size == 4),
        true,
        phi::errors::InvalidArgument("The dims of sin and cos is expected to "
                                     "be 2 or 4, but received %d.",
                                     dims_size));
    if (dims_size == 4) {
      // sin.shape: [1, seq_len, 1, head_dim]
      PADDLE_ENFORCE_EQ((sin_cos_dims[2] == 1),
                        true,
                        phi::errors::InvalidArgument(
                            "The num_heads of sin and cos must be 1."));
    }
    int sin_seq_len_dim = (dims_size) == 4 ? 1 : 0;
    if (position_ids.get_ptr()) {
      PADDLE_ENFORCE_EQ(
          (sin_cos_dims[dims_size - 1] == head_dim &&
           sin_cos_dims[sin_seq_len_dim] >= seq_len),
          true,
          phi::errors::InvalidArgument(
              "The seq_len of sin and cos must be greater than or equal to "
              "this of q. The head_dim of sin and cos must be the same as this "
              "of q."));

      auto position_ids_dims = position_ids.get_ptr()->dims();
      PADDLE_ENFORCE_EQ(position_ids_dims.size(),
                        2,
                        phi::errors::InvalidArgument(
                            "The dims of position_ids is expected to "
                            "be 2, but received %d.",
                            position_ids_dims.size()));

      PADDLE_ENFORCE_EQ(
          (position_ids_dims[0] == batch_size &&
           position_ids_dims[1] == seq_len),
          true,
          phi::errors::InvalidArgument(
              "The batch_size and seq_len of position_ids must be the same as "
              "those of q."));
      using XPUTypeFp16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
      using XPUTypeBf16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;
      if (std::is_same<XPUType, XPUTypeBf16>::value) {
        int ret = xpu::gather<XPUTypeFp16, int64_t>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUTypeFp16*>(sin_cos->data()),
            position_ids->data<int64_t>(),
            reinterpret_cast<XPUTypeFp16*>(sin_cos_data),
            {seq_len, head_dim},
            batch_size * seq_len,
            0);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather");
      } else {
        int ret = xpu::gather<XPUType, int64_t>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(sin_cos->data()),
            position_ids->data<int64_t>(),
            sin_cos_data,
            {seq_len, head_dim},
            batch_size * seq_len,
            0);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather");
      }
    } else {
      int sin_cos_batch_size = (dims_size) == 4 ? sin_cos_dims[0] : 1;
      int ret = xpu::broadcast<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(sin_cos->data()),
          sin_cos_data,
          {sin_cos_batch_size, seq_len, head_dim},
          {batch_size, seq_len, head_dim});
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
    }
  } else {
    int ret = xpu::constant(dev_ctx.x_context(),
                            sin_cos_data,
                            batch_size * seq_len * head_dim,
                            static_cast<XPUType>(0.0f));
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
  }
}

template <typename XPUType, typename Context>
void XPUFusedRotaryHalf(const Context& dev_ctx,
                        const XPUType* in_data,
                        const XPUType* sin_data,
                        const XPUType* cos_data,
                        XPUType* out_data,
                        int64_t batch_size,
                        int64_t seq_len,
                        int64_t num_heads,
                        int64_t head_dim,
                        bool is_bwd = false) {
  auto func = &xpu::rotary_no_freqs_embedding_v2<XPUType>;
  if (is_bwd) {
    func = &xpu::rotary_no_freqs_embedding_v2_grad<XPUType>;
  }

  int ret =
      func(dev_ctx.x_context(),
           in_data,
           sin_data,
           cos_data,
           out_data,
           {batch_size, seq_len, num_heads, head_dim},
           {batch_size, seq_len, 1, head_dim},
           {seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, 1},
           {seq_len * head_dim, head_dim, head_dim, 1});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret,
                              is_bwd ? "rotary_no_freqs_embedding_v2_grad"
                                     : "rotary_no_freqs_embedding_v2");
}
}  // namespace fusion
}  // namespace phi
