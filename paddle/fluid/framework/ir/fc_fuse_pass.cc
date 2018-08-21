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

#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fc_fuse_pass.h"
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
  // make sure the selected MUL op has one input argument is a parameter.
  auto* mul_parameter_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->outputs.size() == 1UL &&
               node->outputs.front()->Op()->Type() == "mul" && node->Var() &&
               node->Var()->Persistable();  // check is a parameter
      },
      "mul_weight" /*name*/);

  auto* mul_tmp_input_var = pattern->NewNode(
      [](Node* node) {
        bool result =
            node->IsVar() && node->outputs.size() >= 1UL && node->Var() &&
            !node->Var()->Persistable();  // this input is not an parameter.
        if (!result) return false;
        // check whether one output is MUL op.
        for (auto* op : node->outputs) {
          if (op->IsOp() && op->Op()->Type() == "mul") return true;
        }
        return false;
      },
      "mul_tmp_var" /*name*/);

  // select a MUL op
  auto* mul_op = pattern->NewNode(
      [](Node* node) {
        return node->IsOp() &&               // start from an Op
               node->Op()->Type() == "mul";  // type is mul
        // the output should be consumed only by one element_add, that check
        // leaves in a Var PDNode.
      },
      "mul" /*name*/);

  // make sure the MUL op's output has only one consumer and links to an
  // ELEMENTWISE_ADD op.
  auto* mul_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() &&                  // starts from a Var
               node->outputs.size() == 1UL &&    // only has one consumer
               node->outputs.front()->IsOp() &&  // check basic logic
               node->Var() &&                    // not a ControlDepVar
               node->outputs.front()->Op()->Type() ==
                   "elementwise_add";  // a very strong validation
      },
      "mul_out");
  // this check is not essential, just to make the corresponding variable Node
  // retrival easier.
  auto* elementwise_add_tmp_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->outputs.size() >= 1UL && node->Var() &&
               VarOutLinksToOp(node, "elementwise_add");
      },
      "elementwise_add_tmpvar");

  // select an ELEMENTWISE_ADD op
  auto* elementwise_add_op = pattern->NewNode(
      [](Node* node) {
        return node->IsOp() && node->Op()->Type() == "elementwise_add";
      },
      "elementwise_add" /*name*/);

  // get the ELEMENTWISE_ADD op's output
  auto* elementwise_add_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->inputs.size() == 1UL && node->Var() &&
               node->inputs.front()->Op()->Type() == "elementwise_add";
      },
      "elementwise_add_out");

  pattern->AddEdge(mul_parameter_var, mul_op);
  pattern->AddEdge(mul_tmp_input_var, mul_op);
  pattern->AddEdge(mul_op, mul_out_var);
  pattern->AddEdge(mul_out_var, elementwise_add_op);
  pattern->AddEdge(elementwise_add_tmp_var, elementwise_add_op);
  pattern->AddEdge(elementwise_add_op, elementwise_add_out_var);
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

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetecter gpd;
  BuildFCPattern(gpd.mutable_pattern());

#define GET_NODE(id)                                             \
  PADDLE_ENFORCE(subgraph.count(gpd.pattern().RetriveNode(#id)), \
                 "pattern has no Node called %s", #id);          \
  auto* id = subgraph.at(gpd.pattern().RetriveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

  auto handler = [&](const GraphPatternDetecter::subgraph_t& subgraph,
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
  };

  gpd(graph.get(), handler);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_fuse_pass, paddle::framework::ir::FCFusePass);
