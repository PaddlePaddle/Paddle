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

#include "paddle/fluid/framework/ir/fc_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

bool VarOutLinksToOp(Node* node, const std::string& op_type) {
  for (auto* out : node->outputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

void BuildFCPattern(PDPattern* pattern) {
  // Create Operators
  auto* mul_op = pattern->NewNode("mul")->assert_is_op("mul");
  auto* elementwise_add_op =
      pattern->NewNode("elementwise_add")->assert_is_op("elementwise_add");
  // Create variables
  // w
  auto* mul_weight_var = pattern->NewNode("mul_weight")
                             ->AsInput()
                             ->assert_is_op_nth_input("mul", "Y", 0);
  // x
  auto* mul_tmp_var = pattern->NewNode("mul_tmp_var")
                          ->AsInput()
                          ->assert_is_op_nth_input("mul", "X", 0);
  // intermediate variable, will be removed in the IR after fuse.
  auto* mul_out_var = pattern->NewNode("mul_out")
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("mul")
                          ->assert_is_op_input("elementwise_add");
  // bias
  auto* elementwise_add_tmp_var = pattern->NewNode("elementwise_add_tmpvar")
                                      ->assert_is_op_input("elementwise_add")
                                      ->AsInput();
  // output
  auto* elementwise_add_out_var = pattern->NewNode("elementwise_add_out")
                                      ->AsOutput()
                                      ->assert_is_op_output("elementwise_add");

  mul_op->LinksFrom({mul_weight_var, mul_tmp_var}).LinksTo({mul_out_var});
  elementwise_add_op->LinksFrom({mul_out_var, elementwise_add_tmp_var})
      .LinksTo({elementwise_add_out_var});
}

// Replace the node `from` in the links to `to`
bool LinksReplace(std::vector<Node*>* links, Node* from, Node* to) {
  for (auto*& n : *links) {
    if (n == from) {
      n = to;
      return true;
    }
  }
  return false;
}

std::unique_ptr<ir::Graph> FCFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fc", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  // BuildFCPattern(gpd.mutable_pattern());
  auto* x = gpd.mutable_pattern()
                ->NewNode("fc_fuse/x")
                ->AsInput()
                ->assert_is_op_input("mul", "X");
  patterns::FC(gpd.mutable_pattern(), "fc_fuse", x, true /*with bias*/);

#define GET_NODE(id)                                                         \
  PADDLE_ENFORCE(subgraph.count(gpd.pattern().RetrieveNode("fc_fuse/" #id)), \
                 "pattern has no Node called %s", #id);                      \
  auto* id = subgraph.at(gpd.pattern().RetrieveNode("fc_fuse/" #id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", "fc_fuse/" #id);

  int found_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FC fuse";
    // Currently, there is no FC op available, so I will just simulate the
    // scenerio.
    // FC's fusion is simple, just op fuse, no need to process the
    // parameters.
    GET_NODE(x);                // x
    GET_NODE(w);                // Y
    GET_NODE(fc_bias);          // bias
    GET_NODE(fc_out);           // Out
    GET_NODE(mul);              // MUL op
    GET_NODE(elementwise_add);  // ELEMENT_ADD op
    GET_NODE(mul_out);          // tmp
#undef GET_NODE

    // Create an FC Node.
    OpDesc desc;
    std::string fc_x_in = x->Name();
    std::string fc_Y_in = w->Name();
    std::string fc_bias_in = fc_bias->Name();
    std::string fc_out_out = fc_out->Name();
    desc.SetInput("Input", std::vector<std::string>({fc_x_in}));
    desc.SetInput("W", std::vector<std::string>({fc_Y_in}));
    desc.SetInput("Bias", std::vector<std::string>({fc_bias_in}));
    desc.SetOutput("Out", std::vector<std::string>({fc_out_out}));
    desc.SetType("fc");
    auto fc_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {mul, elementwise_add, mul_out});

    IR_NODE_LINK_TO(x, fc_node);
    IR_NODE_LINK_TO(w, fc_node);
    IR_NODE_LINK_TO(fc_bias, fc_node);
    IR_NODE_LINK_TO(fc_node, fc_out);

    found_fc_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_fc_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_fuse_pass, paddle::framework::ir::FCFusePass);
