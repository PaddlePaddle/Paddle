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

#include "paddle/fluid/framework/ir/xpu/cross_attention_xpu_fuse_pass.h"

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct CrossAttentionFusePattern : public PatternBase {
  CrossAttentionFusePattern(PDPattern* pattern,
                            const std::string& name_scope,
                            bool with_q_scale);

  // declare operator node's name
  PATTERN_DECL_NODE(q_mul);
  PATTERN_DECL_NODE(k_mul);
  PATTERN_DECL_NODE(v_mul);
  PATTERN_DECL_NODE(q_add);
  PATTERN_DECL_NODE(k_add);
  PATTERN_DECL_NODE(v_add);
  PATTERN_DECL_NODE(reshape_1);
  PATTERN_DECL_NODE(reshape_2);
  PATTERN_DECL_NODE(reshape_3);
  PATTERN_DECL_NODE(transpose_1);
  PATTERN_DECL_NODE(transpose_2);
  PATTERN_DECL_NODE(transpose_3);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(qk_matmul);
  PATTERN_DECL_NODE(qk_add);
  PATTERN_DECL_NODE(qk_softmax);
  PATTERN_DECL_NODE(qkv_matmul);
  PATTERN_DECL_NODE(transpose_4);
  PATTERN_DECL_NODE(reshape_4);

  // declare variable node's name
  PATTERN_DECL_NODE(input_q);
  PATTERN_DECL_NODE(input_kv);
  PATTERN_DECL_NODE(mask);
  PATTERN_DECL_NODE(q_mul_w);
  PATTERN_DECL_NODE(k_mul_w);
  PATTERN_DECL_NODE(v_mul_w);
  PATTERN_DECL_NODE(q_mul_out);
  PATTERN_DECL_NODE(k_mul_out);
  PATTERN_DECL_NODE(v_mul_out);
  PATTERN_DECL_NODE(q_add_bias);
  PATTERN_DECL_NODE(k_add_bias);
  PATTERN_DECL_NODE(v_add_bias);
  PATTERN_DECL_NODE(q_add_out);
  PATTERN_DECL_NODE(k_add_out);
  PATTERN_DECL_NODE(v_add_out);
  PATTERN_DECL_NODE(reshape_1_out);
  PATTERN_DECL_NODE(reshape_2_out);
  PATTERN_DECL_NODE(reshape_3_out);
  PATTERN_DECL_NODE(transpose_1_out);
  PATTERN_DECL_NODE(transpose_2_out);
  PATTERN_DECL_NODE(transpose_3_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(qk_matmul_out);
  PATTERN_DECL_NODE(qk_add_out);
  PATTERN_DECL_NODE(qk_softmax_out);
  PATTERN_DECL_NODE(qkv_matmul_out);
  PATTERN_DECL_NODE(transpose_4_out);
  PATTERN_DECL_NODE(output);

 private:
  bool with_q_scale_{false};
};

CrossAttentionFusePattern::CrossAttentionFusePattern(
    PDPattern* pattern, const std::string& name_scope, bool with_q_scale)
    : PatternBase(pattern, name_scope, name_scope),
      with_q_scale_(with_q_scale) {
  auto* input_q = pattern->NewNode(input_q_repr())
                      ->assert_is_op_input("matmul_v2", "X")
                      ->AsInput();
  auto* input_kv = pattern->NewNode(input_kv_repr())
                       ->assert_is_op_input("matmul_v2", "X")
                       ->AsInput();
  auto* mask = pattern->NewNode(mask_repr())
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->AsInput();
  auto* q_mul_w =
      pattern->NewNode(q_mul_w_repr())->assert_is_op_input("matmul_v2", "Y");
  auto* q_mul = pattern->NewNode(q_mul_repr())->assert_is_op("matmul_v2");
  auto* q_mul_out = pattern->NewNode(q_mul_out_repr())
                        ->assert_is_op_output("matmul_v2", "Out")
                        ->assert_is_op_input("elementwise_add", "X");
  auto* k_mul_w =
      pattern->NewNode(k_mul_w_repr())->assert_is_op_input("matmul_v2", "Y");
  auto* k_mul = pattern->NewNode(k_mul_repr())->assert_is_op("matmul_v2");
  auto* k_mul_out = pattern->NewNode(k_mul_out_repr())
                        ->assert_is_op_output("matmul_v2", "Out")
                        ->assert_is_op_input("elementwise_add", "X");
  auto* v_mul_w =
      pattern->NewNode(v_mul_w_repr())->assert_is_op_input("matmul_v2", "Y");
  auto* v_mul = pattern->NewNode(v_mul_repr())->assert_is_op("matmul_v2");
  auto* v_mul_out = pattern->NewNode(v_mul_out_repr())
                        ->assert_is_op_output("matmul_v2", "Out")
                        ->assert_is_op_input("elementwise_add", "X");
  auto* q_add = pattern->NewNode(q_add_repr())->assert_is_op("elementwise_add");
  auto* q_add_bias = pattern->NewNode(q_add_bias_repr())
                         ->assert_is_op_input("elementwise_add", "Y");
  auto* q_add_out = pattern->NewNode(q_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_is_op_input("reshape2", "X");
  auto* k_add = pattern->NewNode(k_add_repr())->assert_is_op("elementwise_add");
  auto* k_add_bias = pattern->NewNode(k_add_bias_repr())
                         ->assert_is_op_input("elementwise_add", "Y");
  auto* k_add_out = pattern->NewNode(k_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_is_op_input("reshape2", "X");
  auto* v_add = pattern->NewNode(v_add_repr())->assert_is_op("elementwise_add");
  auto* v_add_bias = pattern->NewNode(v_add_bias_repr())
                         ->assert_is_op_input("elementwise_add", "Y");
  auto* v_add_out = pattern->NewNode(v_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_is_op_input("reshape2", "X");
  auto* reshape_1 =
      pattern->NewNode(reshape_1_repr())->assert_is_op("reshape2");
  auto* reshape_1_out = pattern->NewNode(reshape_1_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("transpose2", "X");
  auto* reshape_2 =
      pattern->NewNode(reshape_2_repr())->assert_is_op("reshape2");
  auto* reshape_2_out = pattern->NewNode(reshape_2_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("transpose2", "X");
  auto* reshape_3 =
      pattern->NewNode(reshape_3_repr())->assert_is_op("reshape2");
  auto* reshape_3_out = pattern->NewNode(reshape_3_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("transpose2", "X");
  auto* transpose_1 =
      pattern->NewNode(transpose_1_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });

  auto* transpose_2 =
      pattern->NewNode(transpose_2_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });
  auto* transpose_2_out = pattern->NewNode(transpose_2_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input("matmul_v2", "Y");
  auto* transpose_3 =
      pattern->NewNode(transpose_3_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });
  auto* transpose_3_out = pattern->NewNode(transpose_3_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input("matmul_v2", "Y");
  PDNode* transpose_1_out = nullptr;
  PDNode* scale = nullptr;
  PDNode* scale_out = nullptr;
  if (with_q_scale_) {
    transpose_1_out = pattern->NewNode(transpose_1_out_repr())
                          ->assert_is_op_output("transpose2", "Out")
                          ->assert_is_op_input("scale", "X");
    scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
    scale_out = pattern->NewNode(scale_out_repr())
                    ->assert_is_op_output("scale", "Out")
                    ->assert_is_op_input("matmul_v2", "X");
  } else {
    transpose_1_out = pattern->NewNode(transpose_1_out_repr())
                          ->assert_is_op_output("transpose2", "Out")
                          ->assert_is_op_input("matmul_v2", "X");
  }
  auto* qk_matmul =
      pattern->NewNode(qk_matmul_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output("matmul_v2", "Out")
                            ->assert_is_op_input("elementwise_add", "X");
  auto* qk_add =
      pattern->NewNode(qk_add_repr())->assert_is_op("elementwise_add");
  auto* qk_add_out = pattern->NewNode(qk_add_out_repr())
                         ->assert_is_op_output("elementwise_add", "Out")
                         ->assert_is_op_input("softmax", "X");
  auto* qk_softmax =
      pattern->NewNode(qk_softmax_repr())->assert_is_op("softmax");
  auto* qk_softmax_out = pattern->NewNode(qk_softmax_out_repr())
                             ->assert_is_op_output("softmax", "Out")
                             ->assert_is_op_input("matmul_v2", "X");
  auto* qkv_matmul =
      pattern->NewNode(qkv_matmul_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out = pattern->NewNode(qkv_matmul_out_repr())
                             ->assert_is_op_output("matmul_v2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* transpose_4 =
      pattern->NewNode(transpose_4_repr())->assert_is_op("transpose2");
  auto* transpose_4_out = pattern->NewNode(transpose_4_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input("reshape2", "X");
  auto* reshape_4 =
      pattern->NewNode(reshape_4_repr())->assert_is_op("reshape2");
  auto* output = pattern->NewNode(output_repr())
                     ->AsOutput()
                     ->assert_is_op_output("reshape2", "Out");

  // link nodes
  q_mul->LinksFrom({input_q, q_mul_w}).LinksTo({q_mul_out});
  q_add->LinksFrom({q_mul_out, q_add_bias}).LinksTo({q_add_out});
  reshape_1->LinksFrom({q_add_out}).LinksTo({reshape_1_out});
  transpose_1->LinksFrom({reshape_1_out}).LinksTo({transpose_1_out});
  k_mul->LinksFrom({input_kv, k_mul_w}).LinksTo({k_mul_out});
  k_add->LinksFrom({k_mul_out, k_add_bias}).LinksTo({k_add_out});
  reshape_2->LinksFrom({k_add_out}).LinksTo({reshape_2_out});
  transpose_2->LinksFrom({reshape_2_out}).LinksTo({transpose_2_out});
  if (with_q_scale_) {
    scale->LinksFrom({transpose_1_out}).LinksTo({scale_out});
    qk_matmul->LinksFrom({scale_out, transpose_2_out}).LinksTo({qk_matmul_out});
  } else {
    qk_matmul->LinksFrom({transpose_1_out, transpose_2_out})
        .LinksTo({qk_matmul_out});
  }
  qk_add->LinksFrom({qk_matmul_out, mask}).LinksTo({qk_add_out});
  qk_softmax->LinksFrom({qk_add_out}).LinksTo({qk_softmax_out});
  v_mul->LinksFrom({input_kv, v_mul_w}).LinksTo({v_mul_out});
  v_add->LinksFrom({v_mul_out, v_add_bias}).LinksTo({v_add_out});
  reshape_3->LinksFrom({v_add_out}).LinksTo({reshape_3_out});
  transpose_3->LinksFrom({reshape_3_out}).LinksTo({transpose_3_out});
  qkv_matmul->LinksFrom({qk_softmax_out, transpose_3_out})
      .LinksTo({qkv_matmul_out});
  transpose_4->LinksFrom({qkv_matmul_out}).LinksTo({transpose_4_out});
  reshape_4->LinksFrom({transpose_4_out}).LinksTo({output});
}

}  // namespace patterns

void CrossAttentionXPUFusePass::PrepareQKVWeight(Graph* graph,
                                                 Scope* scope,
                                                 BlockDesc* block,
                                                 Node* w,
                                                 Node** real_w,
                                                 Node** w_max) const {
  phi::DenseTensor w_tensor;
  phi::DenseTensor w_int16_tensor;
  phi::DenseTensor w_max_tensor;

  Assign(scope->Var(w->Name())->Get<phi::DenseTensor>(), &w_tensor);
  CastToFp32(&w_tensor, &w_int16_tensor);
  ConvertWithQuant<float, int16_t>(
      &w_int16_tensor, &w_max_tensor, nullptr, false);

  size_t real_w_hash = HashTensor<int16_t>(w_int16_tensor);
  size_t w_max_hash = HashTensor<float>(w_max_tensor);
  std::string real_w_name = std::to_string(real_w_hash);
  std::string w_max_name = std::to_string(w_max_hash);

  *real_w = FindNodeWithName(graph, real_w_name);

  if (*real_w == nullptr) {
    // Create real_w node
    // Update real_w var_desc in block
    VarDesc real_w_desc(real_w_name);
    real_w_desc.SetPersistable(true);
    real_w_desc.SetShape(common::vectorize(w_int16_tensor.dims()));
    real_w_desc.SetDataType(
        framework::TransToProtoVarType(w_int16_tensor.dtype()));
    *real_w = graph->CreateVarNode(&real_w_desc);
    auto* block_real_w_desc = block->Var(real_w_name);
    block_real_w_desc->SetPersistable(real_w_desc.Persistable());
    block_real_w_desc->SetShape(real_w_desc.GetShape());
    block_real_w_desc->SetDataType(real_w_desc.GetDataType());
    // Create w_max node
    // Update w_max var_desc in block
    VarDesc w_max_desc(w_max_name);
    w_max_desc.SetPersistable(true);
    w_max_desc.SetShape(common::vectorize(w_max_tensor.dims()));
    w_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *w_max = graph->CreateVarNode(&w_max_desc);
    auto* block_w_max_desc = block->Var(w_max_name);
    block_w_max_desc->SetPersistable(w_max_desc.Persistable());
    block_w_max_desc->SetShape(w_max_desc.GetShape());
    block_w_max_desc->SetDataType(w_max_desc.GetDataType());

    // Find real_w/w_max variable in scope
    auto* w_var = scope->FindVar(real_w_name);
    if (w_var == nullptr) {
      // Create qkv_w_intx/qkv_w_max variable/tensor
      Assign(w_int16_tensor,
             scope->Var(real_w_name)->GetMutable<phi::DenseTensor>());
      Assign(w_max_tensor,
             scope->Var(w_max_name)->GetMutable<phi::DenseTensor>());
    } else {
      // Share the same variable
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(w_max_name),
          common::errors::Fatal(
              "w_max(%s) variable should not be nullptr if real_w(%s) "
              "variable is exist.",
              w_max_name,
              real_w_name));
    }
  } else {
    *w_max = FindNodeWithName(graph, w_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *w_max,
        common::errors::Fatal(
            "w_max(%s) variable should not be nullptr if real_w(%s) "
            "variable is exist.",
            w_max_name,
            real_w_name));
  }
}

void CrossAttentionXPUFusePass::PrepareQKVBias(Graph* graph,
                                               Scope* scope,
                                               BlockDesc* block,
                                               Node* q_bias,
                                               Node* k_bias,
                                               Node* v_bias,
                                               Node** real_q_bias,
                                               Node** real_k_bias,
                                               Node** real_v_bias) const {
  phi::DenseTensor* q_bias_tensor;
  phi::DenseTensor* k_bias_tensor;
  phi::DenseTensor* v_bias_tensor;
  phi::DenseTensor q_bias_fp32_tensor;
  phi::DenseTensor k_bias_fp32_tensor;
  phi::DenseTensor v_bias_fp32_tensor;
  q_bias_tensor = scope->Var(q_bias->Name())->GetMutable<phi::DenseTensor>();
  k_bias_tensor = scope->Var(k_bias->Name())->GetMutable<phi::DenseTensor>();
  v_bias_tensor = scope->Var(v_bias->Name())->GetMutable<phi::DenseTensor>();
  CastToFp32(q_bias_tensor, &q_bias_fp32_tensor);
  CastToFp32(k_bias_tensor, &k_bias_fp32_tensor);
  CastToFp32(v_bias_tensor, &v_bias_fp32_tensor);

  size_t q_bias_hash = HashTensor<float>(q_bias_fp32_tensor);
  std::string q_bias_name = std::to_string(q_bias_hash);
  *real_q_bias = FindNodeWithName(graph, q_bias_name);

  size_t k_bias_hash = HashTensor<float>(k_bias_fp32_tensor);
  std::string k_bias_name = std::to_string(k_bias_hash);
  *real_k_bias = FindNodeWithName(graph, k_bias_name);

  size_t v_bias_hash = HashTensor<float>(v_bias_fp32_tensor);
  std::string v_bias_name = std::to_string(v_bias_hash);
  *real_v_bias = FindNodeWithName(graph, v_bias_name);
  if (*real_q_bias == nullptr) {
    // Create q_bias node
    // Update q_bias var_desc in block
    VarDesc q_bias_desc(q_bias_name);
    q_bias_desc.SetPersistable(true);
    q_bias_desc.SetShape(common::vectorize(q_bias_fp32_tensor.dims()));
    q_bias_desc.SetDataType(
        framework::TransToProtoVarType(q_bias_fp32_tensor.dtype()));
    *real_q_bias = graph->CreateVarNode(&q_bias_desc);
    auto* block_q_bias_desc = block->Var(q_bias_name);
    block_q_bias_desc->SetPersistable(q_bias_desc.Persistable());
    block_q_bias_desc->SetShape(q_bias_desc.GetShape());
    block_q_bias_desc->SetDataType(q_bias_desc.GetDataType());
    Assign(q_bias_fp32_tensor,
           scope->Var(q_bias_name)->GetMutable<phi::DenseTensor>());
  }
  if (*real_k_bias == nullptr) {
    // Create k_bias node
    // Update k_bias var_desc in block
    VarDesc k_bias_desc(k_bias_name);
    k_bias_desc.SetPersistable(true);
    k_bias_desc.SetShape(common::vectorize(k_bias_fp32_tensor.dims()));
    k_bias_desc.SetDataType(
        framework::TransToProtoVarType(k_bias_fp32_tensor.dtype()));
    *real_k_bias = graph->CreateVarNode(&k_bias_desc);
    auto* block_k_bias_desc = block->Var(k_bias_name);
    block_k_bias_desc->SetPersistable(k_bias_desc.Persistable());
    block_k_bias_desc->SetShape(k_bias_desc.GetShape());
    block_k_bias_desc->SetDataType(k_bias_desc.GetDataType());
    Assign(k_bias_fp32_tensor,
           scope->Var(k_bias_name)->GetMutable<phi::DenseTensor>());
  }
  if (*real_v_bias == nullptr) {
    // Create v_bias node
    // Update v_bias var_desc in block
    VarDesc v_bias_desc(v_bias_name);
    v_bias_desc.SetPersistable(true);
    v_bias_desc.SetShape(common::vectorize(v_bias_fp32_tensor.dims()));
    v_bias_desc.SetDataType(
        framework::TransToProtoVarType(v_bias_fp32_tensor.dtype()));
    *real_v_bias = graph->CreateVarNode(&v_bias_desc);
    auto* block_v_bias_desc = block->Var(v_bias_name);
    block_v_bias_desc->SetPersistable(v_bias_desc.Persistable());
    block_v_bias_desc->SetShape(v_bias_desc.GetShape());
    block_v_bias_desc->SetDataType(v_bias_desc.GetDataType());
    Assign(v_bias_fp32_tensor,
           scope->Var(v_bias_name)->GetMutable<phi::DenseTensor>());
  }
}

void CrossAttentionXPUFusePass::ApplyCrossAttentionXPUFuse(
    ir::Graph* graph, bool with_q_scale) const {
  GraphPatternDetector gpd;
  patterns::CrossAttentionFusePattern pattern(
      gpd.mutable_pattern(), name_scope_, with_q_scale);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle CrossAttentionXPUFusePass";

    // declare operator node's name
    GET_IR_NODE(q_mul);
    GET_IR_NODE(k_mul);
    GET_IR_NODE(v_mul);
    GET_IR_NODE(q_add);
    GET_IR_NODE(k_add);
    GET_IR_NODE(v_add);
    GET_IR_NODE(reshape_1);
    GET_IR_NODE(reshape_2);
    GET_IR_NODE(reshape_3);
    GET_IR_NODE(transpose_1);
    GET_IR_NODE(transpose_2);
    GET_IR_NODE(transpose_3);
    GET_IR_NODE(scale);
    GET_IR_NODE(qk_matmul);
    GET_IR_NODE(qk_add);
    GET_IR_NODE(qk_softmax);
    GET_IR_NODE(qkv_matmul);
    GET_IR_NODE(transpose_4);
    GET_IR_NODE(reshape_4);

    // declare variable node's name
    GET_IR_NODE(input_q);
    GET_IR_NODE(input_kv);
    GET_IR_NODE(mask);
    GET_IR_NODE(q_mul_w);
    GET_IR_NODE(k_mul_w);
    GET_IR_NODE(v_mul_w);
    GET_IR_NODE(q_mul_out);
    GET_IR_NODE(k_mul_out);
    GET_IR_NODE(v_mul_out);
    GET_IR_NODE(q_add_bias);
    GET_IR_NODE(k_add_bias);
    GET_IR_NODE(v_add_bias);
    GET_IR_NODE(q_add_out);
    GET_IR_NODE(k_add_out);
    GET_IR_NODE(v_add_out);
    GET_IR_NODE(reshape_1_out);
    GET_IR_NODE(reshape_2_out);
    GET_IR_NODE(reshape_3_out);
    GET_IR_NODE(transpose_1_out);
    GET_IR_NODE(transpose_2_out);
    GET_IR_NODE(transpose_3_out);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(qk_matmul_out);
    GET_IR_NODE(qk_add_out);
    GET_IR_NODE(qk_softmax_out);
    GET_IR_NODE(qkv_matmul_out);
    GET_IR_NODE(transpose_4_out);
    GET_IR_NODE(output);

    // generate fuse op
    auto* scope = param_scope();
    auto* block = q_mul->Op()->Block();
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("cross_attention_xpu");

    Node* real_q_w = nullptr;
    Node* q_w_max = nullptr;
    Node* real_k_w = nullptr;
    Node* k_w_max = nullptr;
    Node* real_v_w = nullptr;
    Node* v_w_max = nullptr;
    PrepareQKVWeight(graph, scope, block, q_mul_w, &real_q_w, &q_w_max);
    PrepareQKVWeight(graph, scope, block, k_mul_w, &real_k_w, &k_w_max);
    PrepareQKVWeight(graph, scope, block, v_mul_w, &real_v_w, &v_w_max);

    std::vector<Node*> fc_weight_nodes = {real_q_w, real_k_w, real_v_w};
    std::vector<std::string> fc_weight_names;
    for (auto* node : fc_weight_nodes) {
      if (node) {
        fc_weight_names.push_back(node->Name());
      }
    }
    std::vector<Node*> fc_weight_max_nodes = {q_w_max, k_w_max, v_w_max};
    std::vector<std::string> fc_weight_max_names;
    for (auto* node : fc_weight_max_nodes) {
      if (node) {
        fc_weight_max_names.push_back(node->Name());
      }
    }

    Node* q_add_bias_fp32 = nullptr;
    Node* k_add_bias_fp32 = nullptr;
    Node* v_add_bias_fp32 = nullptr;
    PrepareQKVBias(graph,
                   scope,
                   block,
                   q_add_bias,
                   k_add_bias,
                   v_add_bias,
                   &q_add_bias_fp32,
                   &k_add_bias_fp32,
                   &v_add_bias_fp32);
    std::vector<Node*> fc_bias_nodes = {
        q_add_bias_fp32, k_add_bias_fp32, v_add_bias_fp32};
    std::vector<std::string> fc_bias_names;
    for (auto* node : fc_bias_nodes) {
      if (node) {
        fc_bias_names.push_back(node->Name());
      }
    }

    // set input of fuse_op
    fused_op_desc.SetInput("input_q", {input_q->Name()});
    fused_op_desc.SetInput("input_kv", {input_kv->Name()});
    fused_op_desc.SetInput("fc_weight", fc_weight_names);
    fused_op_desc.SetInput("fc_weight_max", fc_weight_max_names);
    fused_op_desc.SetInput("fc_bias", fc_bias_names);
    fused_op_desc.SetInput("mask", {mask->Name()});

    // set attributes of fuse_op
    if (with_q_scale) {
      float scale_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
      fused_op_desc.SetAttr("alpha", scale_val);
      VLOG(4) << "while with_q_scale, scale_val = " << scale_val;
    } else {
      // in xdnn, 0.0f is default value of NewBaseAttnParam.alpha
      fused_op_desc.SetAttr("alpha", 0.0f);
    }
    fused_op_desc.SetAttr(
        "head_num", static_cast<int>(transpose_1_out->Var()->GetShape()[1]));
    fused_op_desc.SetAttr(
        "head_dim", static_cast<int>(transpose_1_out->Var()->GetShape()[3]));
    // TODO(tianrui): support more out_dtype
    fused_op_desc.SetAttr("out_dtype", input_q->Var()->GetDataType());

    // set output of fuse_op
    VarDesc fused_op_out_max_desc("qkv_max");
    Node* fused_op_out_max = graph->CreateVarNode(&fused_op_out_max_desc);
    fused_op_desc.SetOutput("qkv_max", {"qkv_max"});
    fused_op_desc.SetOutput("qkv", {output->Name()});

    auto* fused_op = graph->CreateOpNode(&fused_op_desc);

    // link input of fuse_op
    IR_NODE_LINK_TO(input_q, fused_op);
    IR_NODE_LINK_TO(input_kv, fused_op);
    for (auto* node : fc_weight_nodes) {
      if (node) {
        IR_NODE_LINK_TO(node, fused_op);
      }
    }
    for (auto* node : fc_weight_max_nodes) {
      if (node) {
        IR_NODE_LINK_TO(node, fused_op);
      }
    }
    for (auto* node : fc_bias_nodes) {
      if (node) {
        IR_NODE_LINK_TO(node, fused_op);
      }
    }
    // link output of fuse_op
    IR_NODE_LINK_TO(fused_op, output);
    IR_NODE_LINK_TO(fused_op, fused_op_out_max);

    // delete useless node
    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(q_mul);
    del_node_set.insert(q_mul_out);
    del_node_set.insert(k_mul);
    del_node_set.insert(k_mul_out);
    del_node_set.insert(v_mul);
    del_node_set.insert(v_mul_out);
    del_node_set.insert(q_add);
    del_node_set.insert(q_add_out);
    del_node_set.insert(k_add);
    del_node_set.insert(k_add_out);
    del_node_set.insert(v_add);
    del_node_set.insert(v_add_out);
    del_node_set.insert(reshape_1);
    del_node_set.insert(reshape_1_out);
    del_node_set.insert(reshape_2);
    del_node_set.insert(reshape_2_out);
    del_node_set.insert(reshape_3);
    del_node_set.insert(reshape_3_out);
    del_node_set.insert(transpose_1);
    del_node_set.insert(transpose_1_out);
    del_node_set.insert(transpose_2);
    del_node_set.insert(transpose_2_out);
    del_node_set.insert(transpose_3);
    del_node_set.insert(transpose_3_out);
    del_node_set.insert(qk_matmul);
    del_node_set.insert(qk_matmul_out);
    del_node_set.insert(qk_add);
    del_node_set.insert(qk_add_out);
    del_node_set.insert(qk_softmax);
    del_node_set.insert(qk_softmax_out);
    del_node_set.insert(qkv_matmul);
    del_node_set.insert(qkv_matmul_out);
    del_node_set.insert(transpose_4);
    del_node_set.insert(transpose_4_out);
    del_node_set.insert(reshape_4);
    if (with_q_scale) {
      del_node_set.insert(scale);
      del_node_set.insert(scale_out);
    }
    GraphSafeRemoveNodes(graph, del_node_set);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void CrossAttentionXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  for (auto with_q_scale : {true, false}) {
    ApplyCrossAttentionXPUFuse(graph, with_q_scale);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cross_attention_xpu_fuse_pass,
              paddle::framework::ir::CrossAttentionXPUFusePass);

REGISTER_PASS_CAPABILITY(cross_attention_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "cross_attention_xpu", 0));
