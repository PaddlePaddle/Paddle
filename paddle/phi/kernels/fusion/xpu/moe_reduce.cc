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

#include <xft/xdnn_plugin.h>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeReduceKernel(const Context& ctx,
                     const DenseTensor& ffn_out,
                     const DenseTensor& expert_scales_float,
                     const DenseTensor& permute_indices_per_token,
                     const DenseTensor& top_k_indices,
                     const paddle::optional<DenseTensor>& ffn2_bias,
                     const bool norm_topk_prob,
                     const float routed_scaling_factor,
                     DenseTensor* output) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const int topk = top_k_indices.dims()[1];
  const int num_rows = ffn_out.dims()[0] / topk;
  const int hidden_size = ffn_out.dims()[1];
  output->Resize({num_rows, hidden_size});
  ctx.template Alloc<T>(output);

  int r = baidu::xpu::xftkernel::xft_moe_ffn_post_fusion<XPUType, int32_t>(
      ctx.x_context(),
      reinterpret_cast<const XPUType*>(ffn_out.data<T>()),
      top_k_indices.data<int32_t>(),
      reinterpret_cast<const XPUType*>(expert_scales_float.data<T>()),
      reinterpret_cast<XPUType*>(output->data<T>()),
      num_rows,
      hidden_size,
      -1,
      topk);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xftkernel::xft_moe_ffn_post_fusion");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(moe_reduce,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeReduceKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
