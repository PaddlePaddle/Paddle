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

#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

int BuildFusion(Graph* graph, const std::string& name_scope, Scope* scope,
                bool with_fc_bias) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Build pattern
  PDNode* x = pattern->NewNode(patterns::PDNodeName(name_scope, "x"))
                  ->assert_is_op_input("mul")
                  ->assert_var_not_persistable();
  patterns::FC fc_pattern(pattern, name_scope);

  // fc_out is a tmp var, will be removed after fuse, so marked as intermediate.
  auto* fc_out = fc_pattern(x, with_fc_bias)->AsIntermediate();
  patterns::LSTM lstm_pattern(pattern, name_scope);
  lstm_pattern(fc_out);

  // Create New OpDesc
  auto lstm_creator = [&](Node* lstm, Node* input, Node* weight_x,
                          Node* weight_h, Node* bias, Node* hidden, Node* cell,
                          Node* xx, Node* fc_bias) {
    OpDesc op_desc;
    op_desc.SetType("fusion_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(X, input);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef SET_IN
    if (with_fc_bias) {
      // Add FC-bias with LSTM-bias and create a new weight
      PADDLE_ENFORCE(scope);
      const std::string& new_bias_var = patterns::UniqueKey("NewBias");
      auto* bias_var = scope->Var(new_bias_var);
      PADDLE_ENFORCE(bias_var);
      auto* bias_tensor = bias_var->GetMutable<framework::LoDTensor>();
      auto* lstm_bias_var = scope->FindVar(bias->Name());
      PADDLE_ENFORCE(lstm_bias_var);
      const auto& lstm_bias_tensor = lstm_bias_var->Get<framework::LoDTensor>();
      bias_tensor->Resize(lstm_bias_tensor.dims());

      auto* fc_bias_var = scope->FindVar(fc_bias->Name());
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();

      auto* data = bias_tensor->mutable_data<float>(platform::CPUPlace());

      for (int i = 0; i < bias_tensor->numel(); i++) {
        data[i] =
            fc_bias_tensor.data<float>()[i] + lstm_bias_tensor.data<float>()[i];
      }
      op_desc.SetInput("Bias", {new_bias_var});
    }

    // Create temp variables.
    const std::string BatchedInput = patterns::UniqueKey("BatchedInput");
    const std::string BatchedCellPreAct =
        patterns::UniqueKey("BatchedCellPreAct");
    const std::string BatchedGate = patterns::UniqueKey("BatchedGate");
    const std::string CheckedCell = patterns::UniqueKey("CheckedCell");

    scope->Var(BatchedInput)->GetMutable<framework::LoDTensor>();
    scope->Var(BatchedCellPreAct)->GetMutable<framework::LoDTensor>();
    scope->Var(BatchedGate)->GetMutable<framework::LoDTensor>();
    scope->Var(CheckedCell)->GetMutable<framework::LoDTensor>();

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetOutput("Cell", {cell->Name()});
    op_desc.SetOutput("XX", {xx->Name()});
    op_desc.SetOutput("BatchedGate", {BatchedGate});
    op_desc.SetOutput("BatchCellPreAct", {BatchedCellPreAct});
    op_desc.SetOutput("BatchedInput", {BatchedInput});
    op_desc.SetOutput("CheckedCell", {CheckedCell});
    op_desc.SetAttr("is_reverse", lstm->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", lstm->Op()->GetAttr("use_peepholes"));
    // TODO(TJ): get from attr
    op_desc.SetAttr("use_seq", true);

    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);
#define OP_SET_OUT(x)                            \
  const std::string x = patterns::UniqueKey(#x); \
  op_desc.SetOutput(#x, {x});                    \
  scope->Var(x)->GetMutable<LoDTensor>()
    OP_SET_OUT(BatchedCell);
    OP_SET_OUT(BatchedHidden);
    OP_SET_OUT(ReorderedH0);
    OP_SET_OUT(ReorderedC0);
#undef OP_SET_OUT

    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(input, op);
    IR_NODE_LINK_TO(weight_x, op);
    IR_NODE_LINK_TO(weight_h, op);
    IR_NODE_LINK_TO(bias, op);
    IR_NODE_LINK_TO(op, hidden);
    return op;
  };

  int fusion_count{0};

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(lstm, lstm, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Cell, Cell, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, Out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      lstm_creator(lstm, subgraph.at(x), w, Weight, Bias, Hidden, Cell, fc_out,
                   fc_bias);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, lstm, elementwise_add, fc_bias});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, mul_out, fc_pattern);
      lstm_creator(lstm, subgraph.at(x), w, Weight, Bias, Hidden, Cell, fc_out,
                   nullptr);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes({mul, lstm});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

ir::Graph* MulLstmFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), false /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

ir::Graph* FCLstmFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), true /*with_fc_bias*/);

  AddStatis(fusion_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mul_lstm_fuse_pass, paddle::framework::ir::MulLstmFusePass);
REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);
