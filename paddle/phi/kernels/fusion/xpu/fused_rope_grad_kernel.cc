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
#define LAUNCH_XPU_FUSED_ROPE_GRAD(T, SCT)                 \
  XPUFusedRopeImpl<T, SCT, Context>(dev_ctx,               \
                                    dout_q,                \
                                    dout_k,                \
                                    dout_v,                \
                                    sin,                   \
                                    cos,                   \
                                    position_ids,          \
                                    use_neox_rotary_style, \
                                    time_major,            \
                                    true,                  \
                                    rotary_emb_base,       \
                                    dq,                    \
                                    dk,                    \
                                    dv);

template <typename T, typename Context>
void FusedRopeGradKernel(const Context& dev_ctx,
                         const paddle::optional<DenseTensor>& sin,
                         const paddle::optional<DenseTensor>& cos,
                         const paddle::optional<DenseTensor>& position_ids,
                         const DenseTensor& dout_q,
                         const paddle::optional<DenseTensor>& dout_k,
                         const paddle::optional<DenseTensor>& dout_v,
                         bool use_neox_rotary_style,
                         bool time_major,
                         float rotary_emb_base,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  dev_ctx.template Alloc<T>(dq);
  if (dout_k) {
    dev_ctx.template Alloc<T>(dk);
  }
  if (dout_v) {
    dev_ctx.template Alloc<T>(dv);
  }
  if (sin && cos) {
    PADDLE_ENFORCE_EQ(sin->dims(),
                      cos->dims(),
                      common::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin->dims(),
                          cos->dims()));
    if (sin->dtype() == phi::DataType::FLOAT32) {
      LAUNCH_XPU_FUSED_ROPE_GRAD(T, float);
    } else {
      PADDLE_ENFORCE_EQ(
          phi::CppTypeToDataType<T>::Type(),
          sin->dtype(),
          common::errors::InvalidArgument(
              "The embedding dtype and sin/cos dtype mismatched."));
      LAUNCH_XPU_FUSED_ROPE_GRAD(T, T);
    }
  } else {
    LAUNCH_XPU_FUSED_ROPE_GRAD(T, float);
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
