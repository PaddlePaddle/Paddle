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
void MoeDispatchKernel(const Context& ctx,
                       const DenseTensor& X,
                       const DenseTensor& gating_output,
                       const int moe_topk,
                       const bool group_moe,
                       const bool topk_only_mode,
                       DenseTensor* permute_input,
                       DenseTensor* sorted_tokens_num_lod,
                       DenseTensor* permute_indices_per_token,
                       DenseTensor* expert_scales_float,
                       DenseTensor* top_k_indices) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  int token_rows = 0;
  auto input_dims = X.dims();
  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = X.dims()[input_dims.size() - 1];
  auto gating_dims = gating_output.dims();
  const int expert_num = gating_dims[gating_dims.size() - 1];
  expert_scales_float->Resize({num_rows, moe_topk});
  top_k_indices->Resize({num_rows, moe_topk});
  permute_input->Resize({moe_topk * num_rows, hidden_size});
  permute_indices_per_token->Resize({moe_topk, num_rows});
  sorted_tokens_num_lod->Resize({expert_num + 1});
  ctx.template Alloc<float>(expert_scales_float);
  ctx.template Alloc<int>(top_k_indices);
  ctx.template Alloc<T>(permute_input);
  ctx.template Alloc<int>(permute_indices_per_token);
  ctx.template Alloc<int>(sorted_tokens_num_lod);
  if (group_moe) {
    // Check if expert_num is divisible by moe_topk, else throw an error
    PADDLE_ENFORCE_EQ(expert_num % moe_topk,
                      0,
                      common::errors::InvalidArgument(
                          "The number of experts (expert_num) "
                          "must be divisible by moe_topk. "
                          "Got expert_num = %d and moe_topk = %d.",
                          expert_num,
                          moe_topk));
  }
  auto index_data = permute_indices_per_token->data<int>();
  int ret = baidu::xpu::xftkernel::xft_moe_group_topk_fusion<float, float, int>(
      ctx.x_context(),
      gating_output.data<float>(),
      expert_scales_float->data<float>(),
      index_data,
      nullptr,
      num_rows,
      expert_num,
      0,
      0,
      moe_topk,  // num of shared expert
      0);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret,
                              "xftkernel::xft_moe_softmax_topk_norm_fusion");
  int* token_nums_per_expert = RAII_GUARD.alloc_l3_or_gm<int>(expert_num);
  ret = baidu::xpu::xftkernel::xft_moe_ffn_pre_sorted<XPUType, int>(
      ctx.x_context(),
      reinterpret_cast<const XPUType*>(X.data<T>()),
      index_data,
      nullptr,
      reinterpret_cast<XPUType*>(permute_input->data<T>()),
      top_k_indices->data<int>(),
      token_nums_per_expert,
      sorted_tokens_num_lod->data<int>(),
      token_rows,
      hidden_size,
      expert_num,
      moe_topk,
      0);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xftkernel::xft_moe_ffn_pre_sorted");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(moe_dispatch,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
