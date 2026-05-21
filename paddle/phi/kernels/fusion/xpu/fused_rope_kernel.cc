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
#define LAUNCH_XPU_FUSED_ROPE(T, SCT)                      \
  XPUFusedRopeImpl<T, SCT, Context>(dev_ctx,               \
                                    q,                     \
                                    k,                     \
                                    v,                     \
                                    sin,                   \
                                    cos,                   \
                                    position_ids,          \
                                    use_neox_rotary_style, \
                                    time_major,            \
                                    false,                 \
                                    rotary_emb_base,       \
                                    out_q,                 \
                                    out_k,                 \
                                    out_v);
template <typename T, typename Context>
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const optional<DenseTensor>& k,
                     const optional<DenseTensor>& v,
                     const optional<DenseTensor>& sin,
                     const optional<DenseTensor>& cos,
                     const optional<DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     float rotary_emb_base,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  int64_t numel = q.numel();
  dev_ctx.template Alloc<T>(out_q);
  if (k) dev_ctx.template Alloc<T>(out_k);
  if (v) dev_ctx.template Alloc<T>(out_v);
  if (numel <= 0) return;

  auto batch_size = q.dims()[0];
  auto num_heads = q.dims()[2];

  int k_num_heads = -1;
  if (k && k->numel() > 0) {
    k_num_heads = k->dims()[2];
    auto k_batch_size = k->dims()[0];
    PADDLE_ENFORCE_LE(
        batch_size,
        k_batch_size,
        common::errors::InvalidArgument("The batch_size of q (%d) must be less "
                                        "than or equal to k's (%d).",
                                        batch_size,
                                        k_batch_size));
  }

  if (v && v->numel() > 0) {
    auto v_num_heads = v->dims()[2];
    if (k_num_heads != -1) {
      PADDLE_ENFORCE_EQ(
          k_num_heads == v_num_heads,
          true,
          common::errors::InvalidArgument(
              "The num_heads of k must be equal to the num_heads of v when v "
              "is not none."
              "But received num_heads of k is %d, num_heads of v is %d",
              k_num_heads,
              v_num_heads));
    }
    PADDLE_ENFORCE_EQ(
        num_heads == v_num_heads ||
            (num_heads != v_num_heads && num_heads % v_num_heads == 0),
        true,
        common::errors::InvalidArgument(
            "The MQA or GQA mode is entered, when the number of heads of qkv "
            "is not exactly the same two by two. This mode requires "
            "num_heads of q to be divisible by k,v."
            "But received num_heads of q is %d, num_heads of k,v is %d",
            num_heads,
            v_num_heads));
    auto v_batch_size = v->dims()[0];
    PADDLE_ENFORCE_LE(
        batch_size,
        v_batch_size,
        common::errors::InvalidArgument("The batch_size of q (%d) must be less "
                                        "than or equal to v's (%d).",
                                        batch_size,
                                        v_batch_size));
  }

  if (sin && cos) {
    PADDLE_ENFORCE_EQ(sin->dims(),
                      cos->dims(),
                      common::errors::InvalidArgument(
                          "The dims of sin and cos must be the same. But "
                          "received sin's dims is {%s}, cos's dims is {%s}.",
                          sin->dims(),
                          cos->dims()));
    // For user provided sin/cos, we use the dtype as is.
    if (sin->dtype() == phi::DataType::FLOAT32) {
      LAUNCH_XPU_FUSED_ROPE(T, float);
    } else {
      PADDLE_ENFORCE_EQ(
          CppTypeToDataType<T>::Type(),
          sin->dtype(),
          common::errors::InvalidArgument(
              "The embedding dtype and sin/cos dtype mismatched."));
      LAUNCH_XPU_FUSED_ROPE(T, T);
    }
  } else {
    // For generated sin/cos, we use fp32 all.
    LAUNCH_XPU_FUSED_ROPE(T, float);
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeKernel,
                   float,
                   phi::float16,
                   phi::bfloat16){};
