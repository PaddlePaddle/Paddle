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
#define LAUNCH_XPU_FUSED_ROPE(XPUType, SinCosType)                 \
  XPUFusedRopeImpl<XPUType, float, Context>(dev_ctx,               \
                                            q,                     \
                                            k,                     \
                                            v,                     \
                                            sin,                   \
                                            cos,                   \
                                            position_ids,          \
                                            use_neox_rotary_style, \
                                            time_major,            \
                                            rotary_emb_base,       \
                                            out_q,                 \
                                            out_k,                 \
                                            out_v);
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

  dev_ctx.template Alloc<T>(out_q);
  if (k) {
    dev_ctx.template Alloc<T>(out_k);
  }
  if (v) {
    dev_ctx.template Alloc<T>(out_v);
  }

  if (sin.get_ptr() && cos.get_ptr()) {
    PADDLE_ENFORCE_EQ(sin.get_ptr()->dims(),
                      cos.get_ptr()->dims(),
                      common::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin.get_ptr()->dims(),
                          cos.get_ptr()->dims()));
    // For user provided sin/cos, we use the dtype as is.
    if (sin->dtype() == phi::DateType::FLOAT32) {
      LAUNCH_XPU_FUSED_ROPE(XPUType, float);
    } else {
      PADDLE_ENFORCE_EQ(
          phi::CppTypeToDtype<T>::Type(),
          sin->dtype(),
          common::errors::InvalidArgument(
              "The embedding dtype and sin/cos dtype mismatched."));
      LAUNCH_XPU_FUSED_ROPE(XPUType, XPUType);
    }
  } else {
    // For generated sin/cos, we use fp32 all.
    LAUNCH_XPU_FUSED_ROPE(XPUType, float);
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
