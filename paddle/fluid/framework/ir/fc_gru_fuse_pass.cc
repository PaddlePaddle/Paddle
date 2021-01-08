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

#include "paddle/fluid/framework/ir/fc_gru_fuse_pass.h"
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("x"))->assert_var_not_persistable();

  // Create pattern.
  patterns::FC fc_pattern(pattern, name_scope);
  auto* fc_out = fc_pattern(x, with_fc_bias, /* with_relu */ false);
  fc_out->AsIntermediate();  // fc_out is a tmp var, will be removed after fuse.

  patterns::GRU gru_pattern(pattern, name_scope);
  gru_pattern(fc_out);

  // Create New OpDesc
  auto gru_creater = [&](Node* gru, Node* x, Node* weight_x, Node* weight_h,
                         Node* bias, Node* hidden, Node* fc_bias) {
    OpDesc op_desc;
    op_desc.SetType("fusion_gru");

#define NEW_NAME(x) name_scope + "/at." #x ".new"
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(X, x);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef SET_IN
    // TODO(grygielski): Add H0 to the pass
    op_desc.SetInput("H0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetAttr("is_reverse", gru->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("origin_mode",
                    gru->Op()->GetAttrIfExists<bool>("origin_mode"));
    // TODO(TJ): This should be a option for infer
    op_desc.SetAttr("use_seq", true);
    op_desc.SetAttr("activation", gru->Op()->GetAttr("activation"));
    op_desc.SetAttr("gate_activation", gru->Op()->GetAttr("gate_activation"));

#define SET_IMTERMEDIATE_OUT(key) op_desc.SetOutput(#key, {NEW_NAME(key)})
    SET_IMTERMEDIATE_OUT(ReorderedH0);
    SET_IMTERMEDIATE_OUT(XX);
    SET_IMTERMEDIATE_OUT(BatchedInput);
    SET_IMTERMEDIATE_OUT(BatchedOut);
#undef SET_IMTERMEDIATE_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    if (with_fc_bias) {
      auto* gru_bias_var = scope->FindVar(bias->Name());
      auto* fc_bias_var = scope->FindVar(fc_bias->Name());
      PADDLE_ENFORCE_NE(
          gru_bias_var, nullptr,
          platform::errors::NotFound("GRU bias var has not been found."));
      PADDLE_ENFORCE_NE(
          fc_bias_var, nullptr,
          platform::errors::NotFound("FC bias var has not been found."));

      auto* gru_bias_tensor = gru_bias_var->GetMutable<LoDTensor>();
      auto* fc_bias_tensor = fc_bias_var->GetMutable<LoDTensor>();
      PADDLE_ENFORCE_EQ(
          gru_bias_tensor->numel(), fc_bias_tensor->numel(),
          platform::errors::PreconditionNotMet(
              "GRU and FC biases have to have equal number of elements."));

      auto gru_bias_data =
          gru_bias_tensor->mutable_data<float>(platform::CPUPlace());
      auto* fc_bias_data = fc_bias_tensor->data<float>();

      // Recompute GRU bias
      for (int i = 0; i < gru_bias_tensor->numel(); ++i) {
        gru_bias_data[i] += fc_bias_data[i];
      }
    }
#undef GET_NODE

#define NEW_IMTERMEDIATE_OUT(key)                \
  VarDesc key(NEW_NAME(key));                    \
  key.SetPersistable(false);                     \
  auto* key##_node = graph->CreateVarNode(&key); \
  IR_NODE_LINK_TO(op, key##_node);

    NEW_IMTERMEDIATE_OUT(ReorderedH0);
    NEW_IMTERMEDIATE_OUT(XX);
    NEW_IMTERMEDIATE_OUT(BatchedInput);
    NEW_IMTERMEDIATE_OUT(BatchedOut);
#undef NEW_NAME
#undef NEW_IMTERMEDIATE_OUT

    IR_NODE_LINK_TO(x, op);
    IR_NODE_LINK_TO(weight_x, op);
    IR_NODE_LINK_TO(weight_h, op);
    IR_NODE_LINK_TO(bias, op);
    IR_NODE_LINK_TO(op, hidden);
    // h0?
    return op;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    auto* x_n = subgraph.at(x);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru, gru, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, gru_pattern);
    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate, BatchGate, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchResetHiddenPrev, BatchResetHiddenPrev,
                              gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchHidden, BatchHidden, gru_pattern);

    // TODO(wilber): Support origin_mode=True.
    if (gru->Op()->GetAttrIfExists<bool>("origin_mode") == true) {
      LOG(INFO) << "fc_gru_fuse_pass not supported when origin_mode=True.";
      return;
    }

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, elementwise_add_out, fc_pattern);

      gru_creater(gru, x_n, w, Weight, Bias, Hidden, fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, gru, elementwise_add, fc_out, mul_out, BatchGate,
           BatchResetHiddenPrev, BatchHidden});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      gru_creater(gru, x_n, w, Weight, Bias, Hidden, nullptr);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, gru, BatchGate, BatchResetHiddenPrev, BatchHidden});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }
#undef GET_NODE

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

void MulGRUFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), false /*with_fc_bias*/);

  AddStatis(fusion_count);
}

void FCGRUFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), true /*with_fc_bias*/);

  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mul_gru_fuse_pass, paddle::framework::ir::MulGRUFusePass);
REGISTER_PASS(fc_gru_fuse_pass, paddle::framework::ir::FCGRUFusePass);
REGISTER_PASS_CAPABILITY(mul_gru_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .EQ("gru", 0)
            .LE("fusion_gru", 1));
REGISTER_PASS_CAPABILITY(fc_gru_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("gru", 0)
            .LE("fusion_gru", 1));
