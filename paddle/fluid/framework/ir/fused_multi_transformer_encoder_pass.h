// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FusedMultiTransformerEncoderPattern : public PatternBase {
  FusedMultiTransformerEncoderPattern(PDPattern* pattern,
                                      const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_multi_transformer_encoder") {}

  PDNode* operator()();

  // Q, K, V path
  PATTERN_DECL_NODE(input0);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(matmul0);
  PATTERN_DECL_NODE(matmul1);
  PATTERN_DECL_NODE(matmul2);
  PATTERN_DECL_NODE(matmul0_w);
  PATTERN_DECL_NODE(matmul1_w);
  PATTERN_DECL_NODE(matmul2_w);
  PATTERN_DECL_NODE(matmul0_out);
  PATTERN_DECL_NODE(matmul1_out);
  PATTERN_DECL_NODE(matmul2_out);
  PATTERN_DECL_NODE(eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd1);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd2);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd1_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd2_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_out);
  PATTERN_DECL_NODE(eltadd1_out);
  PATTERN_DECL_NODE(eltadd2_out);
  PATTERN_DECL_NODE(reshape2_0);
  PATTERN_DECL_NODE(reshape2_1);
  PATTERN_DECL_NODE(reshape2_2);
  PATTERN_DECL_NODE(reshape2_0_out);
  PATTERN_DECL_NODE(reshape2_1_out);
  PATTERN_DECL_NODE(reshape2_2_out);
  PATTERN_DECL_NODE(transpose2_0);
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(transpose2_2);
  PATTERN_DECL_NODE(transpose2_0_out);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(transpose2_2_out);

  // Q, K matmul
  PATTERN_DECL_NODE(matmul_qk);
  PATTERN_DECL_NODE(matmul_qk_out);
  PATTERN_DECL_NODE(eltadd_qk);
  PATTERN_DECL_NODE(eltadd_qk_b);
  PATTERN_DECL_NODE(eltadd_qk_out);
  PATTERN_DECL_NODE(softmax_qk);
  PATTERN_DECL_NODE(softmax_qk_out);

  // QK, V matmul
  PATTERN_DECL_NODE(matmul_qkv);
  PATTERN_DECL_NODE(matmul_qkv_out);
  PATTERN_DECL_NODE(reshape2_qkv);
  PATTERN_DECL_NODE(reshape2_qkv_out);
  PATTERN_DECL_NODE(transpose2_qkv);
  PATTERN_DECL_NODE(transpose2_qkv_out);

  // out linear
  PATTERN_DECL_NODE(matmul_linear);
  PATTERN_DECL_NODE(matmul_linear_w);
  PATTERN_DECL_NODE(matmul_linear_out);
  PATTERN_DECL_NODE(eltadd_linear);
  PATTERN_DECL_NODE(eltadd_linear_b);
  PATTERN_DECL_NODE(eltadd_linear_out);

  // output elementwise_add
  PATTERN_DECL_NODE(eltadd_out)
  PATTERN_DECL_NODE(attention_output);

  // while loop
  PATTERN_DECL_NODE(while0);

  // Feed Forward nodes
  PATTERN_DECL_NODE(ffn_layer_norm);
  PATTERN_DECL_NODE(ffn_layer_norm_scale);
  PATTERN_DECL_NODE(ffn_layer_norm_bias);
  PATTERN_DECL_NODE(ffn_layer_norm_mean);
  PATTERN_DECL_NODE(ffn_layer_norm_variance);
  PATTERN_DECL_NODE(ffn_layer_norm_out);
  PATTERN_DECL_NODE(ffn_matmul0);
  PATTERN_DECL_NODE(ffn_matmul0_w);
  PATTERN_DECL_NODE(ffn_matmul0_out);
  PATTERN_DECL_NODE(ffn_eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_out);
  PATTERN_DECL_NODE(ffn_gelu);
  PATTERN_DECL_NODE(ffn_gelu_out);
  PATTERN_DECL_NODE(ffn_matmul1);
  PATTERN_DECL_NODE(ffn_matmul1_w);
  PATTERN_DECL_NODE(ffn_matmul1_out);
  PATTERN_DECL_NODE(ffn_eltadd1);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_out);

  // output elementwise_add
  PATTERN_DECL_NODE(ffn_eltadd_out)
  PATTERN_DECL_NODE(ffn_output);
};

struct FusedMultiTransformerEncoderFuseQKVPattern : public PatternBase {
  FusedMultiTransformerEncoderFuseQKVPattern(PDPattern* pattern,
                                             const std::string& name_scope)
      : PatternBase(
            pattern, name_scope, "fused_multi_transformer_encoder_fuse_qkv") {}

  PDNode* operator()();

  // Q, K, V path
  PATTERN_DECL_NODE(input0);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(matmul0);
  PATTERN_DECL_NODE(matmul0_w);
  PATTERN_DECL_NODE(matmul0_out);
  PATTERN_DECL_NODE(eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_out);
  PATTERN_DECL_NODE(reshape2_0);
  PATTERN_DECL_NODE(reshape2_0_out);
  PATTERN_DECL_NODE(transpose2_0);
  PATTERN_DECL_NODE(transpose2_0_out);

  PATTERN_DECL_NODE(split0)
  PATTERN_DECL_NODE(split0_q_out)
  PATTERN_DECL_NODE(split0_k_out)
  PATTERN_DECL_NODE(split0_v_out)

  // Q, K matmul
  PATTERN_DECL_NODE(matmul_qk);
  PATTERN_DECL_NODE(matmul_qk_out);
  PATTERN_DECL_NODE(eltadd_qk);
  PATTERN_DECL_NODE(eltadd_qk_b);
  PATTERN_DECL_NODE(eltadd_qk_out);
  PATTERN_DECL_NODE(softmax_qk);
  PATTERN_DECL_NODE(softmax_qk_out);

  // QK, V matmul
  PATTERN_DECL_NODE(matmul_qkv);
  PATTERN_DECL_NODE(matmul_qkv_out);
  PATTERN_DECL_NODE(reshape2_qkv);
  PATTERN_DECL_NODE(reshape2_qkv_out);
  PATTERN_DECL_NODE(transpose2_qkv);
  PATTERN_DECL_NODE(transpose2_qkv_out);

  // while loop
  PATTERN_DECL_NODE(while0);

  // out linear
  PATTERN_DECL_NODE(matmul_linear);
  PATTERN_DECL_NODE(matmul_linear_w);
  PATTERN_DECL_NODE(matmul_linear_out);
  PATTERN_DECL_NODE(eltadd_linear);
  PATTERN_DECL_NODE(eltadd_linear_b);
  PATTERN_DECL_NODE(eltadd_linear_out);

  // output elementwise_add
  PATTERN_DECL_NODE(eltadd_out)
  PATTERN_DECL_NODE(attention_output);

  // Feed Forward nodes
  PATTERN_DECL_NODE(ffn_layer_norm);
  PATTERN_DECL_NODE(ffn_layer_norm_scale);
  PATTERN_DECL_NODE(ffn_layer_norm_bias);
  PATTERN_DECL_NODE(ffn_layer_norm_mean);
  PATTERN_DECL_NODE(ffn_layer_norm_variance);
  PATTERN_DECL_NODE(ffn_layer_norm_out);
  PATTERN_DECL_NODE(ffn_matmul0);
  PATTERN_DECL_NODE(ffn_matmul0_w);
  PATTERN_DECL_NODE(ffn_matmul0_out);
  PATTERN_DECL_NODE(ffn_eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_out);
  PATTERN_DECL_NODE(ffn_gelu);
  PATTERN_DECL_NODE(ffn_gelu_out);
  PATTERN_DECL_NODE(ffn_matmul1);
  PATTERN_DECL_NODE(ffn_matmul1_w);
  PATTERN_DECL_NODE(ffn_matmul1_out);
  PATTERN_DECL_NODE(ffn_eltadd1);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_out);

  // output elementwise_add
  PATTERN_DECL_NODE(ffn_eltadd_out)
  PATTERN_DECL_NODE(ffn_output);
};

struct MultiDevicesFusedMultiTransformerEncoderFuseQKVPattern
    : public PatternBase {
  MultiDevicesFusedMultiTransformerEncoderFuseQKVPattern(
      PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern,
                    name_scope,
                    "multi_devices_fused_multi_transformer_encoder_fuse_qkv") {}

  PDNode* operator()();

  // Q, K, V path
  PATTERN_DECL_NODE(input0);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(c_identity);
  PATTERN_DECL_NODE(c_identity_out);
  PATTERN_DECL_NODE(matmul0);
  PATTERN_DECL_NODE(matmul0_w);
  PATTERN_DECL_NODE(matmul0_out);
  PATTERN_DECL_NODE(eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_out);
  PATTERN_DECL_NODE(reshape2_0);
  PATTERN_DECL_NODE(reshape2_0_out);
  PATTERN_DECL_NODE(transpose2_0);
  PATTERN_DECL_NODE(transpose2_0_out);

  PATTERN_DECL_NODE(split0)
  PATTERN_DECL_NODE(split0_q_out)
  PATTERN_DECL_NODE(split0_k_out)
  PATTERN_DECL_NODE(split0_v_out)

  // Q, K matmul
  PATTERN_DECL_NODE(matmul_qk);
  PATTERN_DECL_NODE(matmul_qk_out);
  PATTERN_DECL_NODE(eltadd_qk);
  PATTERN_DECL_NODE(eltadd_qk_b);
  PATTERN_DECL_NODE(eltadd_qk_out);
  PATTERN_DECL_NODE(softmax_qk);
  PATTERN_DECL_NODE(softmax_qk_out);

  // QK, V matmul
  PATTERN_DECL_NODE(matmul_qkv);
  PATTERN_DECL_NODE(matmul_qkv_out);
  PATTERN_DECL_NODE(reshape2_qkv);
  PATTERN_DECL_NODE(reshape2_qkv_out);
  PATTERN_DECL_NODE(transpose2_qkv);
  PATTERN_DECL_NODE(transpose2_qkv_out);

  // while loop
  PATTERN_DECL_NODE(while0);

  // out linear
  PATTERN_DECL_NODE(matmul_linear);
  PATTERN_DECL_NODE(matmul_linear_w);
  PATTERN_DECL_NODE(matmul_linear_out);
  PATTERN_DECL_NODE(c_allreduce_sum);
  PATTERN_DECL_NODE(c_allreduce_sum_out);
  PATTERN_DECL_NODE(eltadd_linear);
  PATTERN_DECL_NODE(eltadd_linear_b);
  PATTERN_DECL_NODE(eltadd_linear_out);

  // output elementwise_add
  PATTERN_DECL_NODE(eltadd_out)
  PATTERN_DECL_NODE(attention_output);

  // Feed Forward nodes
  PATTERN_DECL_NODE(ffn_layer_norm);
  PATTERN_DECL_NODE(ffn_layer_norm_scale);
  PATTERN_DECL_NODE(ffn_layer_norm_bias);
  PATTERN_DECL_NODE(ffn_layer_norm_mean);
  PATTERN_DECL_NODE(ffn_layer_norm_variance);
  PATTERN_DECL_NODE(ffn_layer_norm_out);
  PATTERN_DECL_NODE(ffn_c_identity);
  PATTERN_DECL_NODE(ffn_c_identity_out);
  PATTERN_DECL_NODE(ffn_matmul0);
  PATTERN_DECL_NODE(ffn_matmul0_w);
  PATTERN_DECL_NODE(ffn_matmul0_out);
  PATTERN_DECL_NODE(ffn_eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd0_out);
  PATTERN_DECL_NODE(ffn_gelu);
  PATTERN_DECL_NODE(ffn_gelu_out);
  PATTERN_DECL_NODE(ffn_matmul1);
  PATTERN_DECL_NODE(ffn_matmul1_w);
  PATTERN_DECL_NODE(ffn_matmul1_out);
  PATTERN_DECL_NODE(ffn_c_allreduce_sum);
  PATTERN_DECL_NODE(ffn_c_allreduce_sum_out);
  PATTERN_DECL_NODE(ffn_eltadd1);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(ffn_eltadd1_out);

  // output elementwise_add
  PATTERN_DECL_NODE(ffn_eltadd_out)
  PATTERN_DECL_NODE(ffn_output);
};
}  // namespace patterns

class FusedMultiTransformerEncoderPass : public FusePassBase {
 public:
  FusedMultiTransformerEncoderPass();
  virtual ~FusedMultiTransformerEncoderPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"fused_multi_transformer_encoder"};

 private:
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope) const;
};

class FusedMultiTransformerEncoderFuseQKVPass : public FusePassBase {
 public:
  FusedMultiTransformerEncoderFuseQKVPass();
  virtual ~FusedMultiTransformerEncoderFuseQKVPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"fused_multi_transformer_encoder_fuse_qkv"};

 private:
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope) const;
};

class MultiDevicesFusedMultiTransformerEncoderFuseQKVPass
    : public FusePassBase {
 public:
  MultiDevicesFusedMultiTransformerEncoderFuseQKVPass();
  virtual ~MultiDevicesFusedMultiTransformerEncoderFuseQKVPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{
      "multi_devices_fused_multi_transformer_encoder_fuse_qkv"};

 private:
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
