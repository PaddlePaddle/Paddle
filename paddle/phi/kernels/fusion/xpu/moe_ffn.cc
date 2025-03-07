// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <xft/core/ctx_manager.h>
#include <xft/core/xft_event.h>
#include <xft/core/xft_tensor.h>
#include <xft/operation/xft_fc_helper.h>
#include <xft/xdnn_plugin.h>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeFFNKernel(const Context& ctx,
                  const DenseTensor& permute_input,
                  const DenseTensor& token_nums_per_expert,
                  const DenseTensor& ffn1_weight,
                  const DenseTensor& ffn2_weight,
                  const paddle::optional<DenseTensor>& ffn1_bias,
                  const paddle::optional<DenseTensor>& ffn1_scale,
                  const paddle::optional<DenseTensor>& ffn2_scale,
                  const std::string& quant_method,
                  DenseTensor* ffn_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  ffn_out->Resize(permute_input.dims());
  ctx.template Alloc<T>(ffn_out);

  const int64_t expanded_active_expert_rows = permute_input.dims()[0];
  const int num_experts = ffn1_weight.dims()[0];
  const int hidden_size = ffn1_weight.dims()[2];
  int64_t inter_size = ffn1_weight.dims()[1];
  if (quant_method == "weight_only_int4") {
    inter_size = inter_size * 2;
  }
  auto ffn1_input = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(const_cast<T*>(permute_input.data<T>())),
      std::array<int64_t, 2>{permute_input.dims()[0], permute_input.dims()[1]});
  auto ffn1_bias_tenosor = baidu::xpu::xft::xftVec<float>(
      ffn1_bias ? const_cast<float*>(ffn1_bias->data<float>()) : nullptr,
      std::array<int64_t, 1>{inter_size});

  XPUType* ffn1_out_paddle = RAII_GUARD.alloc_l3_or_gm<XPUType>(
      expanded_active_expert_rows * inter_size);
  auto ffn1_out_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn1_out_paddle),
      std::array<int64_t, 2>{expanded_active_expert_rows, inter_size});

  auto token_nums_per_expert_tensor = baidu::xpu::xft::xftVec<int32_t>(
      const_cast<int32_t*>(token_nums_per_expert.data<int32_t>()),
      std::array<int64_t, 1>{num_experts});

  auto empty_tensor = baidu::xpu::xft::xftTensor<int32_t, 2>();

  int32_t* sorted_tokens_idx_paddle =
      RAII_GUARD.alloc_l3_or_gm<int32_t>(expanded_active_expert_rows);
  auto sorted_tokens_idx_tensor = baidu::xpu::xft::xftTensor<int32_t, 2>(
      sorted_tokens_idx_paddle,
      std::array<int64_t, 2>{expanded_active_expert_rows, 1});
  int ret = 0;
  if (quant_method == "weight_only_int4") {
    auto ffn1_w_tensor = baidu::xpu::xft::xftMat<int4_t>(
        reinterpret_cast<int4_t*>(
            const_cast<int8_t*>(ffn1_weight.data<int8_t>())),
        nullptr,
        const_cast<float*>(ffn1_scale->data<float>()),
        std::array<int64_t, 2>{num_experts * inter_size, hidden_size});
    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int4_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 int4_wo_int15>(
        ctx.x_context(),
        ffn1_input,
        ffn1_w_tensor,
        ffn1_out_tensor,
        &ffn1_bias_tenosor,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xft_moe_sort_fc_block int4");
  } else {
    auto ffn1_w_tensor = baidu::xpu::xft::xftMat<int8_t>(
        const_cast<int8_t*>(ffn1_weight.data<int8_t>()),
        nullptr,
        const_cast<float*>(ffn1_scale->data<float>()),
        std::array<int64_t, 2>{num_experts * inter_size, hidden_size});
    ret = baidu::xpu::xft::
        xft_moe_sort_fc_block<XPUType, int8_t, XPUType, float, int32_t, float>(
            ctx.x_context(),
            ffn1_input,
            ffn1_w_tensor,
            ffn1_out_tensor,
            &ffn1_bias_tenosor,
            token_nums_per_expert_tensor,
            sorted_tokens_idx_tensor,
            nullptr,
            empty_tensor,
            baidu::xpu::api::Activation_t::LINEAR,
            false,
            true,
            1.0,
            0.0,
            num_experts,
            1,  // topk
            0,
            2,
            1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xft_moe_sort_fc_block int8");
  }

  XPUType* ffn2_input_paddle = RAII_GUARD.alloc_l3_or_gm<XPUType>(
      expanded_active_expert_rows * inter_size / 2);

  auto ffn2_input_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn2_input_paddle),
      std::array<int64_t, 2>{expanded_active_expert_rows, inter_size / 2});
  ret = baidu::xpu::xftkernel::xft_fast_swiglu_add_mul_fusion<XPUType>(
      ctx.x_context(),
      ffn1_out_tensor.data(),
      ffn2_input_tensor.data(),
      expanded_active_expert_rows,
      inter_size,
      nullptr,
      nullptr,
      true,
      nullptr,
      nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xftkernel::xft_fast_swiglu_add_mul_fusion");
  auto ffn2_out_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn_out->data<T>()),
      std::array<int64_t, 2>{ffn_out->dims()[0], ffn_out->dims()[1]});
  if (quant_method == "weight_only_int4") {
    auto ffn2_w_tensor = baidu::xpu::xft::xftMat<int4_t>(
        reinterpret_cast<int4_t*>(
            const_cast<int8_t*>(ffn2_weight.data<int8_t>())),
        nullptr,
        const_cast<float*>(ffn2_scale->data<float>()),
        std::array<int64_t, 2>{num_experts * hidden_size, inter_size / 2});
    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int4_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 int4_wo_int15>(
        ctx.x_context(),
        ffn2_input_tensor,
        ffn2_w_tensor,
        ffn2_out_tensor,
        nullptr,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xft_moe_sort_fc_block int4");
  } else {
    auto ffn2_w_tensor = baidu::xpu::xft::xftMat<int8_t>(
        const_cast<int8_t*>(ffn2_weight.data<int8_t>()),
        nullptr,
        const_cast<float*>(ffn2_scale->data<float>()),
        std::array<int64_t, 2>{num_experts * hidden_size, inter_size / 2});
    ret = baidu::xpu::xft::
        xft_moe_sort_fc_block<XPUType, int8_t, XPUType, float, int32_t, float>(
            ctx.x_context(),
            ffn2_input_tensor,
            ffn2_w_tensor,
            ffn2_out_tensor,
            nullptr,
            token_nums_per_expert_tensor,
            sorted_tokens_idx_tensor,
            nullptr,
            empty_tensor,
            baidu::xpu::api::Activation_t::LINEAR,
            false,
            true,
            1.0,
            0.0,
            num_experts,
            1,  // topk
            0,
            2,
            1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xft_moe_sort_fc_block int8");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(moe_ffn,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeFFNKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
