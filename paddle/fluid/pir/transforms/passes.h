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

#pragma once

#include "paddle/pir/include/pass/pass_registry.h"

USE_PIR_PASS(dead_code_elimination_pass);
USE_PIR_PASS(multihead_matmul_fuse_pass);
USE_PIR_PASS(transpose_flatten_concat_fuse_pass);
USE_PIR_PASS(fused_gemm_epilogue_pass);
USE_PIR_PASS(fused_dropout_add_pass);
USE_PIR_PASS(fused_weight_only_linear_pass);
USE_PIR_PASS(fused_linear_param_grad_add_pass);
USE_PIR_PASS(fuse_allreduce_split_to_reducescatter_pass);
USE_PIR_PASS(inplace_pass);
USE_PIR_PASS(replace_fetch_with_shadow_output_pass);
USE_PIR_PASS(identity_op_clean_pass);
USE_PIR_PASS(map_op_to_another_pass);
USE_PIR_PASS(matmul_scale_fuse_pass);
USE_PIR_PASS(matmul_transpose_fuse_pass);
USE_PIR_PASS(matmul_add_act_fuse_pass);
USE_PIR_PASS(silu_fuse_pass);
USE_PIR_PASS(fc_elementwise_layernorm_fuse_pass);
USE_PIR_PASS(conv2d_bn_fuse_pass);
USE_PIR_PASS(conv2d_add_fuse_pass);
USE_PIR_PASS(conv2d_add_act_fuse_pass);
USE_PIR_PASS(embedding_eltwise_layernorm_fuse_pass);
USE_PIR_PASS(add_norm_fuse_pass);
USE_PIR_PASS(group_norm_silu_fuse_pass);
USE_PIR_PASS(fused_dot_product_attention_pass);
USE_PIR_PASS(fused_flash_attn_pass);
USE_PIR_PASS(remove_redundant_transpose_pass);
USE_PIR_PASS(delete_weight_dequant_linear_op_pass);
USE_PIR_PASS(delete_quant_dequant_linear_op_pass);
USE_PIR_PASS(transfer_layout_pass);
USE_PIR_PASS(fused_rotary_position_embedding_pass);

#ifdef PADDLE_WITH_DNNL
USE_PIR_PASS(depthwise_conv_onednn_pass);
USE_PIR_PASS(squeeze_transpose_onednn_fuse_pass);
USE_PIR_PASS(batch_norm_act_fuse_pass);
USE_PIR_PASS(conv2d_bn_onednn_fuse_pass);
USE_PIR_PASS(conv2d_bias_fuse_pass);
USE_PIR_PASS(conv2d_transpose_bias_fuse_pass);
USE_PIR_PASS(conv3d_bias_fuse_pass);
USE_PIR_PASS(scale_matmul_fuse_pass);
USE_PIR_PASS(reshape_transpose_matmul_fuse_pass);
USE_PIR_PASS(matmul_transpose_reshape_fuse_pass);
USE_PIR_PASS(matmul_elementwise_add_fuse_pass);
USE_PIR_PASS(matmul_activation_fuse_pass);
USE_PIR_PASS(fc_onednn_enable_pass);
USE_PIR_PASS(fc_activation_fuse_pass);
#if defined(PADDLE_WITH_AVX512F) && defined(PADDLE_WITH_MKLML)
USE_PIR_PASS(self_attention_fuse_pass);
#endif
USE_PIR_PASS(softplus_activation_fuse_pass);
USE_PIR_PASS(shuffle_channel_detect_pass);
USE_PIR_PASS(operator_reshape_onednn_fuse_pass);
USE_PIR_PASS(conv_elementwise_add_onednn_fuse_pass);
USE_PIR_PASS(conv_activation_onednn_fuse_pass);
USE_PIR_PASS(conv_concat_activation_onednn_fuse_pass);
USE_PIR_PASS(elementwise_act_onednn_fuse_pass);
USE_PIR_PASS(operator_unsqueeze_onednn_fuse_pass);
USE_PIR_PASS(operator_scale_onednn_fuse_pass);
USE_PIR_PASS(onednn_placement_pass);
#endif

#ifdef PADDLE_WITH_XPU
USE_PIR_PASS(add_layernorm_xpu_fuse_pass);
USE_PIR_PASS(conv2d_bn_xpu_fuse_pass);
USE_PIR_PASS(conv2d_add_xpu_fuse_pass);
USE_PIR_PASS(fc_xpu_fuse_pass);
#endif

#ifdef PADDLE_WITH_CINN
USE_PIR_PASS(convert_MEA_to_FA);
#endif
