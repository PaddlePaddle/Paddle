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

void BuildFCPattern(PDPattern* pattern) {
  // select a MUL op
  auto* mul_op = pattern->NewNode(
      [](Node* node) {
        return node->IsOp() &&                 // start from an Op
               node->inputs.size() == 2UL &&   // it has only two inputs
               node->outputs.size() == 1UL &&  // it has only one output
               node->Op()->Type() == "mul";    // type is mul
      },
      "mul" /*name*/);

  // make sure the MUL op's output has only one consumer and links to an
  // ELEMENTWISE_ADD op.
  auto* mul_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() &&                 // starts from a Var
               node->Var() &&                   // not a ControlDepVar
               node->inputs.size() == 1UL &&    // it has only one input
               node->outputs.size() == 1UL &&   // it has only one output
               node->inputs.front()->IsOp() &&  // check basic logic
               node->inputs.front()->Op()->Type() ==
                   "mul" &&                      // a very strong validation
               node->outputs.front()->IsOp() &&  // check basic logic
               node->outputs.front()->Op()->Type() ==
                   "elementwise_add";  // a very strong validation
      },
      "mul_out");

  // select an ELEMENTWISE_ADD op
  auto* elementwise_add_op = pattern->NewNode(
      [](Node* node) {
        return node->IsOp() && node->inputs.size() == 2UL &&
               node->outputs.size() == 1UL &&
               node->Op()->Type() == "elementwise_add";
      },
      "elementwise_add" /*name*/);
  pattern->AddEdge(mul_op, mul_out_var);
  pattern->AddEdge(mul_out_var, elementwise_add_op);
}

// Replace the node `from` in the links to `to`
bool LinksReplace(std::vector<Node*>* links, Node* from, Node* to) {
  // auto*& -> the reference of pointer, int order to change the element value
  // of vector
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

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetecter gpd;
  BuildFCPattern(gpd.mutable_pattern());

#define GET_NODE(id)                                              \
  PADDLE_ENFORCE(subgraph.count(gpd.pattern().RetrieveNode(#id)), \
                 "pattern has no Node called %s", #id);           \
  auto* id = subgraph.at(gpd.pattern().RetrieveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

  auto handler = [&](const GraphPatternDetecter::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FC fuse";
    // Currently, there is no FC op available, so I will just simulate the
    // scenario.
    // FC's fusion is simple, just op fuse, no need to process the
    // parameters.
    GET_NODE(mul);              // MUL op
    GET_NODE(elementwise_add);  // ELEMENT_ADD op
    GET_NODE(mul_out);          // tmp
#undef GET_NODE

    // Input
    auto* mul_input_var =
        (mul->inputs.front() && !mul->inputs.front()->Var()->Persistable())
            ? mul->inputs.front()
            : mul->inputs.back();
    // W
    auto* mul_weight_var =
        (mul->inputs.front() && mul->inputs.front()->Var()->Persistable())
            ? mul->inputs.front()
            : mul->inputs.back();
    // Bias
    auto* fc_bias_var = (elementwise_add->inputs.front() &&
                         elementwise_add->inputs.front() != mul_out)
                            ? elementwise_add->inputs.front()
                            : elementwise_add->inputs.back();
    // Out
    auto* fc_out_var = elementwise_add->outputs.front();
    // Create an FC Node.
    OpDesc desc;
    std::string fc_x_in = mul_input_var->Name();
    std::string fc_Y_in = mul_weight_var->Name();
    std::string fc_bias_in = fc_bias_var->Name();
    std::string fc_out = fc_out_var->Name();
    desc.SetInput("Input", std::vector<std::string>({fc_x_in}));
    desc.SetInput("W", std::vector<std::string>({fc_Y_in}));
    desc.SetInput("Bias", std::vector<std::string>({fc_bias_in}));
    desc.SetOutput("Out", std::vector<std::string>({fc_out}));
    desc.SetType("fc");
    auto fc_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    fc_node->inputs =
        std::vector<Node*>({mul_input_var, mul_weight_var, fc_bias_var});
    fc_node->outputs.push_back(fc_out_var);

    // Update link relations
    PADDLE_ENFORCE(LinksReplace(&mul_input_var->outputs, mul, fc_node));
    PADDLE_ENFORCE(LinksReplace(&mul_weight_var->outputs, mul, fc_node));
    PADDLE_ENFORCE(
        LinksReplace(&fc_bias_var->outputs, elementwise_add, fc_node));
    PADDLE_ENFORCE(LinksReplace(&fc_out_var->inputs, elementwise_add, fc_node));

    // Drop old nodes
    graph->RemoveNode(mul);
    graph->RemoveNode(elementwise_add);
    graph->RemoveNode(mul_out);  // tmp variable
  };

  gpd(graph.get(), handler);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_fuse_pass, paddle::framework::ir::FCFusePass);
