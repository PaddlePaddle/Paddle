// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/fused_gate_attention_config.h"

namespace phi::fusion {

template <typename T>
void LaunchGateAttentionMergedQKVMatmulForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    DenseTensor* qkv_out,
    const DenseTensor& qkv_weight);

template <typename T>
void LaunchGateAttentionSeparatedQKVMatmulForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* key,
    DenseTensor* query_out,
    DenseTensor* key_out,
    DenseTensor* value_out,
    const DenseTensor& query_weight,
    const DenseTensor& key_weight,
    const DenseTensor& value_weight);

template <typename T>
void LaunchGateAttentionFMHAForward(const GPUContext& dev_ctx,
                                    const DenseTensor* nonbatched_bias,
                                    const DenseTensor* src_mask,
                                    DenseTensor* query_transpose_out,
                                    DenseTensor* key_transpose_out,
                                    DenseTensor* value_transpose_out,
                                    DenseTensor* qkv_transpose_out,
                                    DenseTensor* softmax_out,
                                    DenseTensor* softmax_lse,
                                    DenseTensor* fmha_out,
                                    DenseTensor* gate_out,
                                    funcs::GateAttentionConfig<T>* config);

template <typename T>
void LaunchGateAttentionGatingLinearForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* fmha_out,
    DenseTensor* gate_out,
    bool use_fused_matmul_bias,
    const DenseTensor& gate_weight,
    const DenseTensor& gate_bias);

template <typename T>
void LaunchGateAttentionOutputLinearForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* fmha_or_gate_out,
    DenseTensor* out,
    bool use_fused_matmul_bias,
    const DenseTensor& out_linear_weight,
    const DenseTensor& out_linear_bias);

template <typename T>
void LaunchGateAttentionMergedQKVMatmulBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* qkv_out_grad,
    DenseTensor* query_grad,
    bool use_addto,
    const DenseTensor& qkv_weight,
    DenseTensor* qkv_weight_grad);

template <typename T>
void LaunchGateAttentionSeparatedQKVMatmulBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* key,
    const DenseTensor* query_out_grad,
    const DenseTensor* key_out_grad,
    const DenseTensor* value_out_grad,
    DenseTensor* query_grad,
    DenseTensor* key_grad,
    bool use_addto,
    const DenseTensor& query_weight,
    const DenseTensor& key_weight,
    const DenseTensor& value_weight,
    DenseTensor* query_weight_grad,
    DenseTensor* key_weight_grad,
    DenseTensor* value_weight_grad);

template <typename T>
void LaunchGateAttentionGatingLinearBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* fmha_out,
    const DenseTensor* gate_out_grad,
    DenseTensor* query_grad,
    DenseTensor* fmha_out_grad,
    bool use_fused_matmul_bias,
    const DenseTensor& gate_weight,
    const DenseTensor& gate_bias,
    DenseTensor* gate_weight_grad,
    DenseTensor* gate_bias_grad);

template <typename T>
void LaunchGateAttentionOutputLinearBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* input,
    DenseTensor* input_grad,
    bool use_fused_matmul_bias,
    const DenseTensor& out_grad,
    const DenseTensor& out_linear_weight,
    DenseTensor* out_linear_weight_grad,
    DenseTensor* out_linear_bias_grad);

template <typename T>
void LaunchGateAttentionFMHABackward(const GPUContext& dev_ctx,
                                     const DenseTensor* query_transpose_out,
                                     const DenseTensor* key_transpose_out,
                                     const DenseTensor* value_transpose_out,
                                     const DenseTensor* qkv_transpose_out,
                                     const DenseTensor* softmax_out,
                                     const DenseTensor* softmax_lse,
                                     const DenseTensor* src_mask,
                                     const DenseTensor* nonbatched_bias,
                                     const DenseTensor* fmha_out,
                                     const DenseTensor* fmha_out_grad,
                                     DenseTensor* nonbatched_bias_grad,
                                     funcs::GateAttentionGradConfig<T>* config);

}  // namespace phi::fusion
