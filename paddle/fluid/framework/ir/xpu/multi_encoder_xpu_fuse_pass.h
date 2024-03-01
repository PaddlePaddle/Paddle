// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

/*
step1: fuse single ops to single_encoder_xpu
step2: fuse multi single_encoder_xpu to multi_encoder_xpu

1. step1
Origin subgraph:
        ------------ input_variable*
       |             /      |     \
       |            /       |      \
       |      v_matmul  q_matmul  k_matmul
       |           |        |         |
       |           |        |         |
       |        v_add    q_add      add
       |           |        |         |
       |           |        |         |
       |    v_reshape  q_reshape  k_reshape
       |           |        |         |
       |           |        |         |
       |  v_transpose q_transpose k_transpose
       |          |         |         |
       |          |         |       (scale)
       |          |         \         /
       |          |          qk_matmul
       |          |              |
       |          |              |
       |          |           qk_add
       |          |              |
       |          |              |
       |          |         qk_softmax
       |          |              |
       |          |              |
       |          ---------qkv_matmul_0
       |                         |
       |                         |
       |                  qkv_transpose
       |                         |
       |                         |
       |                    qkv_reshape
       |                         |
       |                         |
       |                    qkv_matmul_1
       |                         |
       |                         |
       |                     qkv_add_0
       |                         |
       |                         |
       ----------------------qkv_add_1
                                |
                                |
                            layer_norm_1
                            /       \
                            |       |
                            |  qkv_matmul_2
                            |       |
                            |       |
                            |   qkv_add_2
                            |       |
                            |       |
                            |    qkv_act
                            |       |
                            |       |
                            |  qkv_matmul_3
                            |       |
                            |       |
                            |   qkv_add_3
                            |       |
                            \       /
                            qkv_add_4
                                |
                            layer_norm

Fused subgraph:
                single_encoder_xpu

2. step2
Origin subgraph:
                       ...
                        |
                single_encoder_xpu
                        |
                (single_encoder_xpu)
                        |
                (single_encoder_xpu)
                        |
                       ...
Fused subgraph:
                multi_encoder_xpu
*/

struct PatternParam {
  std::string act_type;       // "gelu", "relu"
  std::string matmul_type_0;  // "matmul_v2", "matmul", "mul"
  std::string matmul_type_1;  // "matmul_v2", "matmul"
  std::string matmul_type_2;  // "matmul_v2", "matmul"
  bool norm_before;
  bool with_q_scale;
  bool with_mask;
  bool is_smooth_quant;
  std::string relative_type;
};

class MultiEncoderXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplySingleEncoderXPUFuse(ir::Graph* graph,
                                const std::string& act_type,
                                const std::string& matmul_type_0,
                                const std::string& matmul_type_1,
                                const std::string& matmul_type_2,
                                bool norm_before,
                                bool with_q_scale,
                                bool with_mask,
                                bool is_smooth_qunat,
                                const std::string& relative_type) const;

  bool ApplyMultiEncoderXPUFuse(ir::Graph* graph) const;

  // Mask must be fp32 even if model is fp16
  int CastMask(ir::Graph* graph) const;

  // 1. Transpose q_w, k_w, v_w
  // 2. Concat q_w, k_w, v_w
  // 3. Generate qkv_w_max tensor
  // 4. Quant qkv_w to int16/int8 or cast to float16 (local quant)
  void PrepareQKVWeight(
      Graph* graph,
      Scope* scope,
      BlockDesc* block,
      Node* q_w,
      Node* k_w,
      Node* v_w,
      bool enable_int8,
      bool local_quant,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
      Node** qkv_w,
      Node** qkv_w_max,
      Node** qkv_scale_max) const;
  void PrepareInputMax(
      Graph* graph,
      Scope* scope,
      BlockDesc* block,
      std::unordered_map<std::string, std::vector<Node*>>* node_maps,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
      std::vector<Node*>* input_max_nodes,
      std::vector<std::string>* quant_types,
      const std::string* act_type) const;

  // 1. Cast bias to fp32
  // 2. Concat q/k/v bias
  void PrepareQKVBias(Graph* graph,
                      Scope* scope,
                      BlockDesc* block,
                      Node* q_bias,
                      Node* k_bias,
                      Node* v_bias,
                      Node** qkv_bias) const;

  // Iterating all attrs costs too much time.
  // Just provide several cases.
  std::vector<PatternParam> GeneratePatternParams() const;

  const std::string name_scope_{"multi_encoder_xpu_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
