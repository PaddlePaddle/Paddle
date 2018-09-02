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

namespace paddle {
namespace framework {
namespace ir {

std::string GenNodeName(const std::string& prefix, const std::string& name) {
  return prefix + "/" + name;
}

// FC with bias
// op: mul + elementwise_add
// named nodes:
// mul, elementwise_add
// w, mul_out, bias, fc_out
PDNode* FC(PDPattern* pattern, const std::string& name_scope, PDNode* x) {
  // Create Operators
  auto* mul_op = pattern->NewNode(name_scope, "mul")->assert_is_op("mul");
  auto* elementwise_add_op = pattern->NewNode(name_scope, "elementwise_add")
                                 ->assert_is_op("elementwise_add");
  // Create variables
  // w
  auto* mul_weight_var = pattern->NewNode(name_scope, "w")
                             ->AsInput()
                             ->assert_is_op_nth_input("mul", "Y", 0);
  // intermediate variable, will be removed in the IR after fuse.
  auto* mul_out_var = pattern->NewNode(name_scope, "mul_out")
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("mul")
                          ->assert_is_op_input("elementwise_add");
  // bias
  auto* bias = pattern->NewNode(name_scope, "bias")
                   ->assert_is_op_input("elementwise_add")
                   ->AsInput();
  // output
  auto* fc_out = pattern->NewNode(name_scope, "fc_out")
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_add");

  mul_op->LinksFrom({mul_weight_var, x}).LinksTo({mul_out_var});
  elementwise_add_op->LinksFrom({mul_out_var, bias}).LinksTo({fc_out});

  return fc_out;
}

PDNode* LSTM(PDPattern* pattern, const std::string& name_scope, PDNode* x) {
  auto* lstm_op = pattern->NewNode(name_scope, "lstm")->assert_is_op("lstm");
#define NEW_NODE(arg__, io__)                        \
  auto* arg__ = pattern->NewNode(name_scope, #arg__) \
                    ->assert_is_op_##io__("lstm", #arg__);

  NEW_NODE(H0, input);
  NEW_NODE(C0, input);
  NEW_NODE(Weight, input);
  NEW_NODE(Bias, input);

  NEW_NODE(Hidden, output);
  NEW_NODE(Cell, output);
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchCellPreAct, output);

  lstm_op->LinksFrom({H0, C0, Weight, Bias});
  lstm_op->LinksTo({Hidden, Cell, BatchedGate, BatchCellPreAct});
  return Hidden;
}

std::unique_ptr<ir::Graph> FCLstmFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  std::unordered_set<int> fused_ops({// first lstm
                                     13, 15, 16,
                                     // second lstm
                                     23, 25, 26});

  pattern->NewNode([&](Node* x) { return fused_ops.count(x->id()); },
                   "any_node");

  std::unordered_set<Node*> marked_nodes;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {

    auto* id = subgraph.at(gpd.pattern().RetrieveNode("any_node"));
    marked_nodes.insert(id);
  };
  gpd(graph.get(), handler);

  // Create New OpDesc
  auto lstm_creator = [&](int lstm, int input, int weight_x, int weight_h,
                          int bias, int hidden, int cell, int xx) {
#define GET_NODE(x) auto* x##_n = graph->RetriveNode(x);
    GET_NODE(input);
    GET_NODE(weight_x);
    GET_NODE(weight_h);
    GET_NODE(bias);
    GET_NODE(hidden);
    GET_NODE(cell);
    GET_NODE(xx);
    GET_NODE(lstm);

    OpDesc op_desc;
    op_desc.SetType("fusion_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__##_n->Name()});
    SET_IN(X, input);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef GET_NODE
#undef SET_IN

    VLOG(4) << "hidden_n: " << hidden_n->Name();
    VLOG(4) << "cell: " << cell_n->Name();
    VLOG(4) << "xx: " << xx_n->Name();

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden_n->Name()});
    op_desc.SetOutput("Cell", {cell_n->Name()});
    op_desc.SetOutput("XX", {xx_n->Name()});
    op_desc.SetOutput("BatchedGate", {"blstm_0.tmp_2"});
    op_desc.SetOutput("BatchCellPreAct", {"blstm_1.tmp_2"});
    op_desc.SetAttr("is_reverse", lstm_n->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", false);
    auto* op = graph->CreateOpNode(&op_desc);

#define LINK_TO(a, b)      \
  a->outputs.push_back(b); \
  b->inputs.push_back(a);
    LINK_TO(input_n, op);
    LINK_TO(weight_x_n, op);
    LINK_TO(weight_h_n, op);
    LINK_TO(bias_n, op);
    LINK_TO(op, hidden_n);
#undef LINK_TO
    return op;

  };

  lstm_creator(16, 12, 14, 18, 17, 22, 21, 19);
  lstm_creator(26, 12, 24, 28, 27, 32, 31, 29);

  // remove all the nodes

  for (auto* node : marked_nodes) {
    graph->RemoveNode(const_cast<Node*>(node));
  }

  for (auto* node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (marked_nodes.count(*it)) {
        it = const_cast<Node*>(node)->inputs.erase(it);
      } else
        it++;
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (marked_nodes.count(*it)) {
        it = const_cast<Node*>(node)->outputs.erase(it);
      } else
        it++;
    }
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);
