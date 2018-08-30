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
  auto* elementwise_add_op = pattern->NewNode("elementwise_add")
                                 ->assert_op_has_n_inputs("elementwise_add", 2);
  // Create variables
  // w
  auto* mul_weight_var =
      pattern->NewNode("mul_weight")->AsInput()->assert_is_op_input("mul");
  // x
  auto* mul_tmp_var =
      pattern->NewNode("mul_tmp_var")->AsInput()->assert_is_op_input("mul");
  // intermediate variable, will be removed in the IR after fuse.
  auto* mul_out_var = pattern->NewNode("mul_out")
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("mul")
                          ->assert_is_op_input("elementwise_add");
  // bias
  auto* elementwise_add_tmp_var =
      pattern->NewNode("elementwise_add_tmpvar")
          ->AsInput()
          ->assert_is_op_nth_input("elementwise_add", "X", 1);
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
  BuildFCPattern(gpd.mutable_pattern());

#define GET_NODE(id)                                              \
  PADDLE_ENFORCE(subgraph.count(gpd.pattern().RetrieveNode(#id)), \
                 "pattern has no Node called %s", #id);           \
  auto* id = subgraph.at(gpd.pattern().RetrieveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

  int found_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FC fuse";
    // Currently, there is no FC op available, so I will just simulate the
    // scenerio.
    // FC's fusion is simple, just op fuse, no need to process the
    // parameters.
    GET_NODE(mul_tmp_var);             // x
    GET_NODE(mul_weight);              // Y
    GET_NODE(elementwise_add_tmpvar);  // bias
    GET_NODE(elementwise_add_out);     // Out
    GET_NODE(mul);                     // MUL op
    GET_NODE(elementwise_add);         // ELEMENT_ADD op
    GET_NODE(mul_out);                 // tmp
#undef GET_NODE

    // Create an FC Node.
    OpDesc desc;
    std::string fc_x_in = mul_tmp_var->Name();
    std::string fc_Y_in = mul_weight->Name();
    std::string fc_bias_in = elementwise_add_tmpvar->Name();
    std::string fc_out = elementwise_add_out->Name();
    desc.SetInput("Input", std::vector<std::string>({fc_x_in}));
    desc.SetInput("W", std::vector<std::string>({fc_Y_in}));
    desc.SetInput("Bias", std::vector<std::string>({fc_bias_in}));
    desc.SetOutput("Out", std::vector<std::string>({fc_out}));
    desc.SetType("fc");
    auto fc_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    fc_node->inputs =
        std::vector<Node*>({mul_tmp_var, mul_weight, elementwise_add_tmpvar});
    fc_node->outputs.push_back(elementwise_add_out);

    // Update link relatons
    PADDLE_ENFORCE(LinksReplace(&mul_tmp_var->outputs, mul, fc_node));
    PADDLE_ENFORCE(LinksReplace(&mul_weight->outputs, mul, fc_node));
    PADDLE_ENFORCE(LinksReplace(&elementwise_add_tmpvar->outputs,
                                elementwise_add, fc_node));
    PADDLE_ENFORCE(
        LinksReplace(&elementwise_add_out->inputs, elementwise_add, fc_node));

    // Drop old nodes
    graph->RemoveNode(mul);
    graph->RemoveNode(elementwise_add);
    graph->RemoveNode(mul_out);  // tmp variable

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
