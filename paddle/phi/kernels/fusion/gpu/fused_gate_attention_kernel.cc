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
void FusedGateAttentionOpKernel(const Context& dev_ctx,
                                const DenseTensor& query_in,
                                const optional<DenseTensor>& key_in,
                                const optional<DenseTensor>& query_weight_in,
                                const optional<DenseTensor>& key_weight_in,
                                const optional<DenseTensor>& value_weight_in,
                                const optional<DenseTensor>& qkv_weight_in,
                                const optional<DenseTensor>& nonbatched_bias_in,
                                const DenseTensor& src_mask_in,
                                const optional<DenseTensor>& gate_weight_in,
                                const optional<DenseTensor>& gate_bias_in,
                                const DenseTensor& out_linear_weight_in,
                                const DenseTensor& out_linear_bias_in,
                                bool has_gating,
                                bool merge_qkv,
                                bool use_flash_attn,
                                DenseTensor* query_transpose_out,
                                DenseTensor* key_transpose_out,
                                DenseTensor* value_transpose_out,
                                DenseTensor* qkv_transpose_out,
                                DenseTensor* softmax_out,
                                DenseTensor* softmax_lse,
                                DenseTensor* fmha_out,
                                DenseTensor* gate_out,
                                DenseTensor* out) {
  const auto* query = &query_in;
  const auto* key = key_in.get_ptr();
  const auto* query_weight = query_weight_in.get_ptr();
  const auto* qkv_weight = qkv_weight_in.get_ptr();
  const auto* nonbatched_bias = nonbatched_bias_in.get_ptr();

  constexpr bool use_fused_matmul_bias = true;
  funcs::AllocWithDebugInfo<T>(dev_ctx, "fmha_out", fmha_out);
  if (has_gating) {
    funcs::AllocWithDebugInfo<T>(dev_ctx, "gate_out", gate_out);
  }
  funcs::AllocWithDebugInfo<T>(dev_ctx, "out", out);

  funcs::GateAttentionConfig<T> config(dev_ctx,
                                       query,
                                       key,
                                       query_weight,
                                       qkv_weight,
                                       merge_qkv,
                                       has_gating,
                                       use_flash_attn);

  if (merge_qkv) {
    PADDLE_ENFORCE_EQ(
        !key || query == key || query->data<T>() == key->data<T>(),
        true,
        errors::InvalidArgument("key is expected to be nullptr or the same as "
                                "query, but received key=%p, query=%p.",
                                key,
                                query));

    DenseTensor* qkv_out = config.GetQKVOut();
    LaunchGateAttentionMergedQKVMatmulForward<T>(
        dev_ctx, config, query, qkv_out, qkv_weight_in.get());

    if (config.CanUseFlashAttn()) {
      qkv_transpose_out->Resize({3,
                                 config.batch_size,
                                 config.seq_len_m,
                                 config.seq_len_r,
                                 config.num_heads,
                                 config.head_dim});
    }
    funcs::AllocWithDebugInfo<T>(
        dev_ctx, "qkv_transpose_out", qkv_transpose_out);
  } else {
    DenseTensor* query_out = config.GetQueryOut();
    DenseTensor* key_out = config.GetKeyOut();
    DenseTensor* value_out = config.GetValueOut();
    LaunchGateAttentionSeparatedQKVMatmulForward<T>(dev_ctx,
                                                    config,
                                                    query,
                                                    key,
                                                    query_out,
                                                    key_out,
                                                    value_out,
                                                    query_weight_in.get(),
                                                    key_weight_in.get(),
                                                    value_weight_in.get());

    funcs::AllocWithDebugInfo<T>(
        dev_ctx, "q_transpose_out", query_transpose_out);
    funcs::AllocWithDebugInfo<T>(dev_ctx, "k_transpose_out", key_transpose_out);
    funcs::AllocWithDebugInfo<T>(
        dev_ctx, "v_transpose_out", value_transpose_out);
  }

  if (!config.CanUseFlashAttn()) {
    funcs::AllocWithDebugInfo<T>(dev_ctx, "softmax_out", softmax_out);
  }
  LaunchGateAttentionFMHAForward<T>(dev_ctx,
                                    nonbatched_bias,
                                    &src_mask_in,
                                    query_transpose_out,
                                    key_transpose_out,
                                    value_transpose_out,
                                    qkv_transpose_out,
                                    softmax_out,
                                    softmax_lse,
                                    fmha_out,
                                    gate_out,
                                    &config);

  if (has_gating) {
    LaunchGateAttentionGatingLinearForward<T>(dev_ctx,
                                              config,
                                              query,
                                              fmha_out,
                                              gate_out,
                                              use_fused_matmul_bias,
                                              gate_weight_in.get(),
                                              gate_bias_in.get());
  }

  DenseTensor* fmha_or_gate_out = has_gating ? gate_out : fmha_out;
  LaunchGateAttentionOutputLinearForward<T>(dev_ctx,
                                            config,
                                            fmha_or_gate_out,
                                            out,
                                            use_fused_matmul_bias,
                                            out_linear_weight_in,
                                            out_linear_bias_in);
}

}  // namespace phi::fusion

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_gate_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionOpKernel,
                   float,
                   phi::float16,
                   phi::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_gate_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionOpKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
#endif
