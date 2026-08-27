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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fused_gate_attention_config.h"
#include "paddle/phi/kernels/fusion/gpu/fused_gate_attention_kernel_launch.h"
#include "paddle/utils/optional.h"

namespace phi::fusion {

template <typename T, typename Context>
void FusedGateAttentionGradKernel(
    const Context& dev_ctx,
    const DenseTensor& query_in,
    const optional<DenseTensor>& key_in,
    const optional<DenseTensor>& query_weight_in,
    const optional<DenseTensor>& key_weight_in,
    const optional<DenseTensor>& value_weight_in,
    const optional<DenseTensor>& qkv_weight_in,
    const optional<DenseTensor>& nonbatched_bias_in,
    const optional<DenseTensor>& src_mask_in,
    const optional<DenseTensor>& gate_weight_in,
    const optional<DenseTensor>& gate_bias_in,
    const DenseTensor& out_linear_weight_in,
    const DenseTensor& out_linear_bias_in,
    const optional<DenseTensor>& query_transpose_out_in,
    const optional<DenseTensor>& key_transpose_out_in,
    const optional<DenseTensor>& value_transpose_out_in,
    const optional<DenseTensor>& qkv_transpose_out_in,
    const optional<DenseTensor>& softmax_out_in,
    const optional<DenseTensor>& softmax_lse_in,
    const DenseTensor& fmha_out_in,
    const optional<DenseTensor>& gate_out_in,
    const DenseTensor& out_grad_in,
    bool has_gating,
    bool merge_qkv,
    bool use_flash_attn,
    DenseTensor* query_grad,
    DenseTensor* key_grad,
    DenseTensor* query_weight_grad,
    DenseTensor* key_weight_grad,
    DenseTensor* value_weight_grad,
    DenseTensor* qkv_weight_grad,
    DenseTensor* nonbatched_bias_grad,
    DenseTensor* gate_weight_grad,
    DenseTensor* gate_bias_grad,
    DenseTensor* out_linear_weight_grad,
    DenseTensor* out_linear_bias_grad) {
  const auto* query = &query_in;
  const auto* key = key_in.get_ptr();
  const auto* query_weight = query_weight_in.get_ptr();
  const auto* qkv_weight = qkv_weight_in.get_ptr();
  const auto* query_transpose_out = query_transpose_out_in.get_ptr();
  const auto* key_transpose_out = key_transpose_out_in.get_ptr();
  const auto* value_transpose_out = value_transpose_out_in.get_ptr();
  const auto* qkv_transpose_out = qkv_transpose_out_in.get_ptr();
  const auto* softmax_out = softmax_out_in.get_ptr();
  const auto* softmax_lse = softmax_lse_in.get_ptr();
  const auto* gate_out = gate_out_in.get_ptr();

  constexpr bool use_fused_matmul_bias = true;
  funcs::AllocWithDebugInfo<T>(dev_ctx, "query_grad", query_grad);

  funcs::GateAttentionGradConfig<T> config(dev_ctx,
                                           query,
                                           key,
                                           query_weight,
                                           qkv_weight,
                                           merge_qkv,
                                           has_gating,
                                           use_flash_attn);

  DenseTensor fmha_out_grad;
  fmha_out_grad.Resize(config.gate_out_dims);
  funcs::AllocWithDebugInfo<T>(dev_ctx, "fmha_out_grad", &fmha_out_grad);
  if (has_gating) {
    DenseTensor gate_out_grad;
    gate_out_grad.Resize(config.gate_out_dims);
    funcs::AllocWithDebugInfo<T>(dev_ctx, "gate_out_grad", &gate_out_grad);
    LaunchGateAttentionOutputLinearBackward<T>(dev_ctx,
                                               config,
                                               gate_out,
                                               &gate_out_grad,
                                               use_fused_matmul_bias,
                                               out_grad_in,
                                               out_linear_weight_in,
                                               out_linear_weight_grad,
                                               out_linear_bias_grad);
    LaunchGateAttentionGatingLinearBackward<T>(dev_ctx,
                                               config,
                                               query,
                                               &fmha_out_in,
                                               &gate_out_grad,
                                               query_grad,
                                               &fmha_out_grad,
                                               use_fused_matmul_bias,
                                               gate_weight_in.get(),
                                               gate_bias_in.get(),
                                               gate_weight_grad,
                                               gate_bias_grad);
  } else {
    LaunchGateAttentionOutputLinearBackward<T>(dev_ctx,
                                               config,
                                               &fmha_out_in,
                                               &fmha_out_grad,
                                               use_fused_matmul_bias,
                                               out_grad_in,
                                               out_linear_weight_in,
                                               out_linear_weight_grad,
                                               out_linear_bias_grad);
  }

  if (nonbatched_bias_grad) {
    funcs::AllocWithDebugInfo<T>(
        dev_ctx, "nonbatched_bias_grad", nonbatched_bias_grad);
  }
  LaunchGateAttentionFMHABackward<T>(dev_ctx,
                                     query_transpose_out,
                                     key_transpose_out,
                                     value_transpose_out,
                                     qkv_transpose_out,
                                     softmax_out,
                                     softmax_lse,
                                     src_mask_in.get_ptr(),
                                     nonbatched_bias_in.get_ptr(),
                                     &fmha_out_in,
                                     &fmha_out_grad,
                                     nonbatched_bias_grad,
                                     &config);

  const bool use_addto = has_gating;
  if (merge_qkv) {
    DenseTensor* qkv_out_grad = config.GetQKVOutGrad();
    LaunchGateAttentionMergedQKVMatmulBackward<T>(dev_ctx,
                                                  config,
                                                  query,
                                                  qkv_out_grad,
                                                  query_grad,
                                                  use_addto,
                                                  qkv_weight_in.get(),
                                                  qkv_weight_grad);
  } else {
    if (key_grad) {
      funcs::AllocWithDebugInfo<T>(dev_ctx, "key_grad", key_grad);
    }
    DenseTensor* query_out_grad = config.GetQueryOutGrad();
    DenseTensor* key_out_grad = config.GetKeyOutGrad();
    DenseTensor* value_out_grad = config.GetValueOutGrad();
    LaunchGateAttentionSeparatedQKVMatmulBackward<T>(dev_ctx,
                                                     config,
                                                     query,
                                                     key,
                                                     query_out_grad,
                                                     key_out_grad,
                                                     value_out_grad,
                                                     query_grad,
                                                     key_grad,
                                                     use_addto,
                                                     query_weight_in.get(),
                                                     key_weight_in.get(),
                                                     value_weight_in.get(),
                                                     query_weight_grad,
                                                     key_weight_grad,
                                                     value_weight_grad);
  }
}

}  // namespace phi::fusion

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_gate_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_gate_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
#endif
