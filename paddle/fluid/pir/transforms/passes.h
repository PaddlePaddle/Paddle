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
USE_PIR_PASS(inplace_pass);
USE_PIR_PASS(replace_fetch_with_shadow_output_pass);
USE_PIR_PASS(identity_op_clean_pass);
USE_PIR_PASS(map_op_to_another_pass);
USE_PIR_PASS(matmul_scale_fuse_pass);
USE_PIR_PASS(matmul_transpose_fuse_pass);
USE_PIR_PASS(fc_fuse_pass);
USE_PIR_PASS(silu_fuse_pass);
USE_PIR_PASS(fc_elementwise_layernorm_fuse_pass);
USE_PIR_PASS(conv2d_bn_fuse_pass);
USE_PIR_PASS(conv2d_add_fuse_pass);
USE_PIR_PASS(conv2d_add_act_fuse_pass);
USE_PIR_PASS(embedding_eltwise_layernorm_fuse_pass);
USE_PIR_PASS(fused_dot_product_attention_pass);
USE_PIR_PASS(flash_attn_fuse_pass);

#ifdef PADDLE_WITH_DNNL
USE_PIR_PASS(batch_norm_act_fuse_pass);
USE_PIR_PASS(conv2d_bias_fuse_pass);
USE_PIR_PASS(conv2d_transpose_bias_fuse_pass);
USE_PIR_PASS(conv3d_bias_fuse_pass);
USE_PIR_PASS(matmul_elementwise_add_fuse_pass);
USE_PIR_PASS(conv_elementwise_add_mkldnn_fuse_pass);
#endif
