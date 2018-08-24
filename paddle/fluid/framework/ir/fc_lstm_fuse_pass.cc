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

void BuildFcLstmPattern(PDPattern* pattern) {
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
               node->inputs.front()->Op()->Type() == "elementwise_add" &&
               node->outputs.size() == 1UL &&
               node->outputs.front()->Op()->Type() == "lstm";
      },
      "elementwise_add_out");

  auto* lstm_weight_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->outputs.size() == 1UL &&
               node->outputs.front()->Op()->Type() == "lstm" && node->Var() &&
               node->Var()->Persistable();  // check is a parameter
      },
      "lstm_weight" /*name*/);

  auto* lstm_bias_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->outputs.size() == 1UL &&
               node->outputs.front()->Op()->Type() == "lstm" && node->Var() &&
               node->Var()->Persistable();  // check is a parameter
      },
      "lstm_bias");

  auto* lstm_op = pattern->NewNode(
      [](Node* node) { return node->IsOp() && node->Op()->Type() == "lstm"; },
      "lstm");

  auto* lstm_hidden_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->inputs.size() == 1UL && node->Var() &&
               node->inputs.front()->Op()->Type() == "lstm";
      },
      "lstm_hidden_out");

  auto* lstm_cell_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->inputs.size() == 1UL && node->Var() &&
               node->inputs.front()->Op()->Type() == "lstm";
      },
      "lstm_cell_out");

  auto* lstm_batchedgate_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->inputs.size() == 1UL && node->Var() &&
               node->inputs.front()->Op()->Type() == "lstm";
      },
      "lstm_batchedgate_out");

  auto* lstm_batch_cell_preact_out_var = pattern->NewNode(
      [](Node* node) {
        return node->IsVar() && node->inputs.size() == 1UL && node->Var() &&
               node->inputs.front()->Op()->Type() == "lstm";
      },
      "lstm_batch_cell_preact_out");

  pattern->AddEdge(mul_parameter_var, mul_op);
  pattern->AddEdge(mul_tmp_input_var, mul_op);
  pattern->AddEdge(mul_op, mul_out_var);
  pattern->AddEdge(mul_out_var, elementwise_add_op);
  pattern->AddEdge(elementwise_add_tmp_var, elementwise_add_op);
  pattern->AddEdge(elementwise_add_op, elementwise_add_out_var);

  pattern->AddEdge(elementwise_add_out_var, lstm_op);
  pattern->AddEdge(lstm_weight_var, lstm_op);
  pattern->AddEdge(lstm_bias_var, lstm_op);
  pattern->AddEdge(lstm_op, lstm_hidden_out_var);
  pattern->AddEdge(lstm_op, lstm_cell_out_var);
  pattern->AddEdge(lstm_op, lstm_batchedgate_out_var);
  pattern->AddEdge(lstm_op, lstm_batch_cell_preact_out_var);
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

std::unique_ptr<ir::Graph> FcLstmFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetecter gpd;
  BuildFcLstmPattern(gpd.mutable_pattern());

#define GET_NODE(id)                                             \
  PADDLE_ENFORCE(subgraph.count(gpd.pattern().RetriveNode(#id)), \
                 "pattern has no Node called %s", #id);          \
  auto* id = subgraph.at(gpd.pattern().RetriveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

  auto handler = [&](const GraphPatternDetecter::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FCLSTMfuse";
    GET_NODE(mul_tmp_var);
    GET_NODE(mul_weight);
    GET_NODE(elementwise_add_tmpvar);
    GET_NODE(elementwise_add_out);
    GET_NODE(mul);
    GET_NODE(elementwise_add);
    GET_NODE(mul_out);
    GET_NODE(lstm_weight);
    GET_NODE(lstm_bias);
    GET_NODE(lstm);
    GET_NODE(lstm_hidden_out);
    GET_NODE(lstm_cell_out);
    GET_NODE(lstm_batchedgate_out);
    GET_NODE(lstm_batch_cell_preact_out);
#undef GET_NODE

    // Create an FCLSTM Node.
    OpDesc desc;
    // input
    std::string lstm_X = mul_tmp_var->Name();
    std::string lstm_WeightX = mul_weight->Name();
    std::string lstm_WeightH = lstm_weight->Name();
    std::string lstm_Bias = lstm_bias->Name();

    // output
    std::string lstm_Hidden = lstm_hidden_out->Name();
    std::string lstm_Cell = lstm_cell_out->Name();
    std::string lstm_BatchedGate = lstm_batchedgate_out->Name();
    std::string lstm_BatchCellPreAct = lstm_batch_cell_preact_out->Name();
    // temp
    std::string elementwise_add_out_name = elementwise_add_out->Name();

    // set inputs
    desc.SetInput("X", std::vector<std::string>({lstm_X}));
    desc.SetInput("WeightX", std::vector<std::string>({lstm_WeightX}));
    desc.SetInput("WeightH", std::vector<std::string>({lstm_WeightH}));
    desc.SetInput("Bias", std::vector<std::string>({lstm_Bias}));

    // set ouptuts
    desc.SetOutput("Hidden", std::vector<std::string>({lstm_Hidden}));
    desc.SetOutput("Cell", std::vector<std::string>({lstm_Cell}));
    desc.SetOutput("BatchedGate", std::vector<std::string>({lstm_BatchedGate}));
    desc.SetOutput("BatchCellPreAct",
                   std::vector<std::string>({lstm_BatchCellPreAct}));

    desc.SetOutput("XX", std::vector<std::string>({elementwise_add_out_name}));
    desc.SetType("fusion_lstm");

    OpDesc* lstm_desc = lstm->Op();

    // set attr for fusion_lstm
    std::string candidate_activation =
        boost::get<std::string>(lstm_desc->GetAttr("candidate_activation"));
    std::string cell_activation =
        boost::get<std::string>(lstm_desc->GetAttr("cell_activation"));
    std::string gate_activation =
        boost::get<std::string>(lstm_desc->GetAttr("gate_activation"));
    bool is_reverse = boost::get<bool>(lstm_desc->GetAttr("is_reverse"));
    bool use_peepholes = boost::get<bool>(lstm_desc->GetAttr("use_peepholes"));

    desc.SetAttr("candidate_activation", candidate_activation);
    desc.SetAttr("cell_activation", cell_activation);
    desc.SetAttr("gate_activation", gate_activation);
    desc.SetAttr("is_reverse", is_reverse);
    desc.SetAttr("use_peepholes", use_peepholes);

    auto fusion_lstm_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    fusion_lstm_node->inputs =
        std::vector<Node*>({mul_tmp_var, mul_weight, lstm_weight, lstm_bias});
    fusion_lstm_node->outputs =
        std::vector<Node*>({lstm_hidden_out, lstm_cell_out,
                            lstm_batchedgate_out, lstm_batch_cell_preact_out});

    // Update link relatons
    PADDLE_ENFORCE(LinksReplace(&mul_tmp_var->outputs, mul, fusion_lstm_node));
    PADDLE_ENFORCE(LinksReplace(&mul_weight->outputs, mul, fusion_lstm_node));
    PADDLE_ENFORCE(LinksReplace(&lstm_weight->outputs, lstm, fusion_lstm_node));
    PADDLE_ENFORCE(LinksReplace(&lstm_bias->outputs, lstm, fusion_lstm_node));
    //  PADDLE_ENFORCE(LinksReplace(&utputs,
    //                            elementwise_add, fc_node));
    PADDLE_ENFORCE(
        LinksReplace(&lstm_hidden_out->inputs, lstm, fusion_lstm_node));
    PADDLE_ENFORCE(
        LinksReplace(&lstm_cell_out->inputs, lstm, fusion_lstm_node));
    PADDLE_ENFORCE(
        LinksReplace(&lstm_batchedgate_out->inputs, lstm, fusion_lstm_node));
    PADDLE_ENFORCE(LinksReplace(&lstm_batch_cell_preact_out->inputs, lstm,
                                fusion_lstm_node));

    // Drop old nodes
    graph->RemoveNode(mul);
    graph->RemoveNode(elementwise_add);
    graph->RemoveNode(mul_out);
    graph->RemoveNode(elementwise_add_out);
    graph->RemoveNode(lstm);
  };

  gpd(graph.get(), handler);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FcLstmFusePass);
