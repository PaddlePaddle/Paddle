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

namespace paddle {
namespace framework {
namespace ir {

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       Scope* scope, bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::FC fc_pattern(pattern, name_scope);
  patterns::GRU gru_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("x"))->assert_var_not_persistable();

  auto* fc_out = fc_pattern(x, with_fc_bias);
  fc_out->AsIntermediate();  // fc_out is a tmp var, will be removed after fuse.
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
    if (with_fc_bias) {
      op_desc.SetInput("Bias", {NEW_NAME(bias) + bias->Name()});
    } else {
      SET_IN(Bias, bias);
    }
#undef SET_IN
    op_desc.SetInput("H0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetAttr("is_reverse", gru->Op()->GetAttr("is_reverse"));
    // TODO(TJ): This should be a option for infer
    op_desc.SetAttr("use_seq", true);

#define SET_IMTERMEDIATE_OUT(key) op_desc.SetOutput(#key, {NEW_NAME(key)})
    SET_IMTERMEDIATE_OUT(ReorderedH0);
    SET_IMTERMEDIATE_OUT(XX);
    SET_IMTERMEDIATE_OUT(BatchedInput);
    SET_IMTERMEDIATE_OUT(BatchedOut);
#undef SET_IMTERMEDIATE_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);
    PADDLE_ENFORCE(scope);
    if (with_fc_bias) {
      // Fusion GRU bias = fcbias + grubias
      auto* fusion_bias_var = scope->Var(NEW_NAME(bias) + bias->Name());
      auto* out_bias_tensor =
          fusion_bias_var->GetMutable<framework::LoDTensor>();
      PADDLE_ENFORCE(fusion_bias_var);
      auto* gru_bias_var = scope->FindVar(bias->Name());
      auto* fc_bias_var = scope->FindVar(fc_bias->Name());
      PADDLE_ENFORCE(gru_bias_var);
      PADDLE_ENFORCE(fc_bias_var);
      const auto& gru_bias_tenosr = gru_bias_var->Get<framework::LoDTensor>();
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();
      // new bias = fc bias + gru bias
      out_bias_tensor->Resize(gru_bias_tenosr.dims());
      auto* data = out_bias_tensor->mutable_data<float>(platform::CPUPlace());
      for (int i = 0; i < out_bias_tensor->numel(); i++) {
        data[i] =
            fc_bias_tensor.data<float>()[i] + gru_bias_tenosr.data<float>()[i];
      }
    }
#undef GET_NODE

#define NEW_IMTERMEDIATE_OUT(key) \
  scope->Var(NEW_NAME(key))->GetMutable<framework::LoDTensor>()
    NEW_IMTERMEDIATE_OUT(ReorderedH0);
    NEW_IMTERMEDIATE_OUT(XX);
    NEW_IMTERMEDIATE_OUT(BatchedInput);
    NEW_IMTERMEDIATE_OUT(BatchedOut);
#undef NEW_NAME
#undef NEW_IMTERMEDIATE_OUT

    IR_NODE_LINK_TO(x, op);
    IR_NODE_LINK_TO(weight_x, op);
    IR_NODE_LINK_TO(weight_h, op);
    IR_NODE_LINK_TO(bias, op);  // actually should link to new bias if have
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
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, Out, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru, gru, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, gru_pattern);
    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate, BatchGate, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchResetHiddenPrev, BatchGate, gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchHidden, BatchGate, gru_pattern);

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);

      gru_creater(gru, x_n, w, Weight, Bias, Hidden, fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, gru, elementwise_add, fc_bias, fc_out, mul_out, BatchGate,
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
