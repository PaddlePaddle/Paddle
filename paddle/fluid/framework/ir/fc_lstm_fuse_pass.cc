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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

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
    op_desc.SetOutput("BatchedInput", {"blstm_0.tmp_2"});
    op_desc.SetAttr("is_reverse", lstm_n->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", false);

#define TMP_NAME(x) "at.new.tmp." #x
#define OP_SET_OUT(x) op_desc.SetOutput(#x, {TMP_NAME(x)})
    OP_SET_OUT(BatchedCell);
    OP_SET_OUT(BatchedHidden);
    OP_SET_OUT(ReorderedH0);
    OP_SET_OUT(ReorderedC0);
#undef OP_SET_OUT
    auto* op = graph->CreateOpNode(&op_desc);

    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);

#define TMP_NEW(x) scope->Var(TMP_NAME(x))->GetMutable<LoDTensor>()
    TMP_NEW(BatchedCell);
    TMP_NEW(BatchedHidden);
    TMP_NEW(ReorderedH0);
    TMP_NEW(ReorderedC0);

#undef TMP_NEW
#undef TMP_NAME

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
      } else {
        it++;
      }
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (marked_nodes.count(*it)) {
        it = const_cast<Node*>(node)->outputs.erase(it);
      } else {
        it++;
      }
    }
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);
