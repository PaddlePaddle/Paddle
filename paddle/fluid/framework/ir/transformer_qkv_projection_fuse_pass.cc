// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/transformer_qkv_projection_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

PDNode* TansformerQKVProjectionPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("mul");

  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("mul");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("mul");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("mul");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("mul");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("mul");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("mul");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");

  return eltadd2_out_var;
}

}  // namespace patterns

TansformerQKVProjectionFusePass::TansformerQKVProjectionFusePass() {
  AddOpCompat(OpCompat("mul"))
      .AddInput("X")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddInput("Y")  // the shape shoule be (N*H, N*H)
      .IsTensor()
      .End()
      .AddOutput("Out")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(2)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      // in bias, shape is (B, S, N*H),
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      .AddInput("Y")
      // in bias, shape is (N*H)
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      // in bias, shape is (B, S, N*H)
      // in biasqk, shape is (B, H, S, S)
      .AddOutput("Out")
      .IsTensor()
      .End()
      // in bias, it equal to 2
      // in biasqk, it equal to -1 or 0
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();
}

int TansformerQKVProjectionFusePass::BuildFusion(Graph* graph,
                                                 const std::string& name_scope,
                                                 Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::TansformerQKVProjectionPattern qkv_pattern(pattern, name_scope);

  qkv_pattern();
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* input0,  // shared input of three mul ops
      Node* mul0_w, Node* mul1_w, Node* mul2_w,           // mul weights
      Node* mul0, Node* mul1, Node* mul2,                 // mul ops
      Node* mul0_out, Node* mul1_out, Node* mul2_out,     // mul outputs
      Node* eltadd0_b, Node* eltadd1_b, Node* eltadd2_b,  // bias
      Node* eltadd0, Node* eltadd1, Node* eltadd2,        // bias ops
      Node* eltadd0_out, Node* eltadd1_out,
      Node* eltadd2_out) {  // bias outputs
    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor = scope->FindVar(mul0_w->Name())->GetMutable<LoDTensor>();
    auto* wk_tensor = scope->FindVar(mul1_w->Name())->GetMutable<LoDTensor>();
    auto* wv_tensor = scope->FindVar(mul2_w->Name())->GetMutable<LoDTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<LoDTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<LoDTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<LoDTensor>();

    auto* wq_data = wq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wk_data = wk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wv_data = wv_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bq_data = bq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bk_data = bk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bv_data = bv_tensor->mutable_data<float>(platform::CPUPlace());

    auto combined_w_dims =
        framework::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    auto combined_bias_dims = framework::make_ddim({3, bq_tensor->dims()[0]});

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = mul0_w->Var();
    combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, bq_tensor->dims()[0]});
    combined_bias_desc->SetPersistable(true);

    framework::LoDTensor tmp_combined_w_tensor;
    tmp_combined_w_tensor.Resize(combined_w_dims);
    auto* tmp_combined_w_data =
        tmp_combined_w_tensor.mutable_data<float>(platform::CPUPlace());

    std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
    int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
    // Combine the three fc weights together.
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (3 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          tmp_combined_w_data[out_index] = w_vec[j][in_index];
        }
      }
    }

    wq_tensor->Resize(combined_w_dims);
    auto* new_combined_w_data =
        wq_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_combined_w_data, tmp_combined_w_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({mul1_w->Name(), mul2_w->Name()});

    // Combine the three bias together.
    framework::LoDTensor tmp_combined_bias_tensor;
    tmp_combined_bias_tensor.Resize(combined_bias_dims);
    auto* tmp_combined_bias_data =
        tmp_combined_bias_tensor.mutable_data<float>(platform::CPUPlace());

    size_t bias_size = bq_tensor->numel();
    memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(tmp_combined_bias_data + bias_size, bk_data,
           sizeof(float) * bias_size);
    memcpy(tmp_combined_bias_data + 2 * bias_size, bv_data,
           sizeof(float) * bias_size);

    bq_tensor->Resize(combined_bias_dims);
    auto* new_combined_bias_data =
        bq_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_combined_bias_data, tmp_combined_bias_data,
           sizeof(float) * bq_tensor->numel());

    scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});

    // Define fused QKV op
    OpDesc fused_qkv_op_desc(mul0->Op()->Block());
    fused_qkv_op_desc.SetType("fc");

    fused_qkv_op_desc.SetInput("Input", {input0->Name()});
    fused_qkv_op_desc.SetInput("W", {mul0_w->Name()});
    fused_qkv_op_desc.SetInput("Bias", {eltadd0_b->Name()});

    fused_qkv_op_desc.SetOutput("Out", {mul0_out->Name()});

    auto* mul0_op_desc = mul0->Op();
    auto* mul1_op_desc = mul1->Op();
    auto* mul2_op_desc = mul2->Op();
    if (mul0_op_desc->HasAttr("enable_int8")) {
      multihead_op_desc.SetAttr("enable_int8",
                                mul0_op_desc->GetAttr("enable_int8"));
      // all mul op has same input.
      fused_qkv_op_desc.SetAttr("Input_scale",
                                mul0_op_desc->GetAttr("X_scale"));
      auto weight_scale0 = BOOST_GET_CONST(
          std::vector<float>, mul0_op_desc->GetAttr("weight_scale"));
      auto weight_scale1 = BOOST_GET_CONST(
          std::vector<float>, mul1_op_desc->GetAttr("weight_scale"));
      auto weight_scale2 = BOOST_GET_CONST(
          std::vector<float>, mul2_op_desc->GetAttr("weight_scale"));
      auto weight_max = std::max(weight_scale0, weight_scale1);
      weight_max = std::max(weight_max, weight_scale2);
      fused_qkv_op_desc.SetAttr("weight_scale", weight_max);

      auto* add0_op_desc = eltadd0->Op();
      auto* add1_op_desc = eltadd1->Op();
      auto* add2_op_desc = eltadd2->Op();
      if (add0_op_desc->HasAttr("out_threshold")) {
        auto out_scale0 =
            BOOST_GET_CONST(float, add0_op_desc->GetAttr("out_threshold"));
        auto out_scale1 =
            BOOST_GET_CONST(float, add1_op_desc->GetAttr("out_threshold"));
        auto out_scale2 =
            BOOST_GET_CONST(float, add2_op_desc->GetAttr("out_threshold"));
        auto out_scale_max = std::max(out_scale0, out_scale1);
        out_scale_max = std::max(out_scale_max, out_scale2);
        fused_qkv_op_desc.SetAttr("fc_out_threshold", out_scale_max);
      }
    }

    // Define split op
    OpDesc split_op_desc(mul0->Op()->Block());
    split_op_desc.SetType("split");

    split_op_desc.SetInput(
        "X", {mul0_out->Name()});  // [batch_size, seq_length, 3, hidden_out]
    split_op_desc.SetOutput(
        "Out", {eltadd0_out->Name(), eltadd1_out->Name(),
                eltadd2_out->Name()});  // [batch_size, seq_length, hidden_out]
    split_op_desc.SetAttr("axis", 2);
    split_op_desc.SetAttr("num", 3);
    split_op_desc.SetAttr("squeeze", true);

    auto* fused_qkv = graph->CreateOpNode(&fused_qkv_op_desc);
    auto* split = graph->CreateOpNode(&split_op_desc);

    IR_NODE_LINK_TO(input0, fused_qkv);
    IR_NODE_LINK_TO(mul0_w, fused_qkv);
    IR_NODE_LINK_TO(eltadd0_b, fused_qkv);

    IR_NODE_LINK_TO(fused_qkv, mul0_out);
    IR_NODE_LINK_TO(mul0_out, split);
    IR_NODE_LINK_TO(split, eltadd0_out);
    IR_NODE_LINK_TO(split, eltadd1_out);
    IR_NODE_LINK_TO(split, eltadd2_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "Op compat check in transfomer_qkv_projection_fuse_pass failed.";
      return;
    }
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, qkv_pattern);

    // If weights or biases in qkv's fc are shared by multiple multihead_matmul
    // patterns, we do not support this kind of fusion, this pass will not take
    // effect.
    bool is_fc_params_shared =
        mul0_w->outputs.size() > 1 || mul1_w->outputs.size() > 1 ||
        mul2_w->outputs.size() > 1 || eltadd0_b->outputs.size() > 1 ||
        eltadd1_b->outputs.size() > 1 || eltadd2_b->outputs.size() > 1;
    if (is_fc_params_shared) {
      return;
    }

    fuse_creater(input0,                        // shared input of three mul ops
                 mul0_w, mul1_w, mul2_w,        // mul weights
                 mul0, mul1, mul2,              // mul ops
                 mul0_out, mul1_out, mul2_out,  // mul outputs
                 eltadd0_b, eltadd1_b, eltadd2_b,         // bias
                 eltadd0, eltadd1, eltadd2,               // bias ops
                 eltadd0_out, eltadd1_out, eltadd2_out);  // bias outputs)

    // Mark nodes to be removed
    std::unordered_set<const Node*> marked_nodes({
        // input0
        // mul0_w, keep it for storing combined weights
        mul1_w, mul2_w, mul0, mul1, mul2, mul0_out, mul1_out, mul2_out,
        // eltadd0_b, keep it for storing combined bias
        eltadd1_b, eltadd2_b, eltadd0, eltadd1, eltadd2,
        // eltadd0_out,
        // eltadd1_out,
        // eltadd2_out,
    });
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void TansformerQKVProjectionFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multiheadMatmul pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kTransformerQKVProjectionFusePass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(transfomer_qkv_projection_fuse_pass,
              paddle::framework::ir::TansformerQKVProjectionFusePass);
REGISTER_PASS_CAPABILITY(transfomer_qkv_projection_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1));
