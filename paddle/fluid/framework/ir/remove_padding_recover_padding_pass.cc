// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/remove_padding_recover_padding_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
void SkipLayernorm::operator()() {
  // Create nodes for skip_layernorm.
  auto* skip_layernorm_x = pattern->NewNode(skip_layernorm_x_repr())
                               ->assert_is_op_input("skip_layernorm", "X");
  auto* skip_layernorm_y = pattern->NewNode(skip_layernorm_y_repr())
                               ->assert_is_op_input("skip_layernorm", "Y");
  auto* skip_layernorm_op = pattern->NewNode(skip_layernorm_op_repr())
                                ->assert_is_op("skip_layernorm");
  auto* skip_layernorm_out = pattern->NewNode(skip_layernorm_out_repr())
                                 ->assert_is_op_output("skip_layernorm", "Out");

  // Add links for skip_layernorm op.
  skip_layernorm_op->LinksFrom({skip_layernorm_x, skip_layernorm_y})
      .LinksTo({skip_layernorm_out});
}

void MultiheadMatmul::operator()() {
  // Create nodes for multihead_matmul.
  auto* multihead_matmul_input =
      pattern->NewNode(multihead_matmul_input_repr())
          ->assert_is_op_input("multihead_matmul", "Input");
  auto* multihead_matmul_op = pattern->NewNode(multihead_matmul_op_repr())
                                  ->assert_is_op("multihead_matmul");
  auto* multihead_matmul_out =
      pattern->NewNode(multihead_matmul_out_repr())
          ->assert_is_op_output("multihead_matmul", "Out");

  // Add links for multihead_matmul op.
  multihead_matmul_op->LinksFrom({multihead_matmul_input})
      .LinksTo({multihead_matmul_out});
}

void Fc::operator()() {
  // Create nodes for fc.
  auto* fc_input =
      pattern->NewNode(fc_input_repr())->assert_is_op_input("fc", "Input");
  auto* fc_op = pattern->NewNode(fc_op_repr())->assert_is_op("fc");
  auto* fc_out =
      pattern->NewNode(fc_out_repr())->assert_is_op_output("fc", "Out");

  // Add links for fc op.
  fc_op->LinksFrom({fc_input}).LinksTo({fc_out});
}

void Activation::operator()() {
  // Create nodes for activation.
  std::unordered_set<std::string> activation_ops{"relu", "sigmoid", "tanh"};
  auto* activation_input = pattern->NewNode(activation_input_repr())
                               ->assert_is_ops_input(activation_ops);
  auto* activation_op =
      pattern->NewNode(activation_op_repr())->assert_is_ops(activation_ops);
  auto* activation_out = pattern->NewNode(activation_out_repr())
                             ->assert_is_ops_output(activation_ops);

  // Add links for activation op.
  activation_op->LinksFrom({activation_input}).LinksTo({activation_out});
}
}  // namespace patterns

void RemovePaddingRecoverPaddingPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  int found_subgraph_count = 0;

  // Create an remove_padding op node
  auto insert_remove_padding_op = [&](Node* input_node, Node* op_node) {
    OpDesc remove_padding;
    std::string remove_padding_out_name =
        input_node->Name() + ".remove_padding";
    VarDesc remove_padding_out(remove_padding_out_name);

    // remove_padding_op
    remove_padding.SetType("remove_padding");

    // input
    remove_padding.SetInput("Input", {input_node->Name()});

    // output
    remove_padding.SetOutput("Out", {remove_padding_out_name});

    auto remove_padding_op_node = graph->CreateOpNode(&remove_padding);
    auto remove_padding_out_node = graph->CreateVarNode(&remove_padding_out);

    // replace link
    for (size_t i = 0; i < input_node->outputs.size(); ++i) {
      if (input_node->outputs[i] == op_node) {
        input_node->outputs[i] = remove_padding_op_node;
        remove_padding_op_node->inputs.push_back(input_node);
      }
    }

    // link node
    IR_NODE_LINK_TO(remove_padding_op_node, remove_padding_out_node);

    // replace link
    for (size_t i = 0; i < op_node->inputs.size(); ++i) {
      if (op_node->inputs[i] == input_node) {
        op_node->inputs[i] = remove_padding_out_node;
        remove_padding_out_node->outputs.push_back(op_node);
      }
    }

    // rename
    op_node->Op()->RenameInput(input_node->Name(),
                               remove_padding_out_node->Name());
  };

  // Create an remove_padding op node
  auto insert_recover_padding_op = [&](Node* op_node, Node* out_node) {
    OpDesc recover_padding;
    std::string recover_padding_input_name =
        out_node->Name() + ".recover_padding";
    VarDesc recover_padding_input(recover_padding_input_name);

    // recover_padding_op
    recover_padding.SetType("recover_padding");

    // input
    recover_padding.SetInput("Input", {recover_padding_input_name});

    // output
    recover_padding.SetOutput("Out", {out_node->Name()});

    auto recover_padding_op_node = graph->CreateOpNode(&recover_padding);
    auto recover_padding_input_node =
        graph->CreateVarNode(&recover_padding_input);

    // replace link
    for (size_t i = 0; i < op_node->outputs.size(); ++i) {
      if (op_node->outputs[i] == out_node) {
        op_node->outputs[i] = recover_padding_input_node;
        recover_padding_input_node->inputs.push_back(op_node);
      }
    }

    // link node
    IR_NODE_LINK_TO(recover_padding_input_node, recover_padding_op_node);

    // replace link
    for (size_t i = 0; i < out_node->inputs.size(); ++i) {
      if (out_node->inputs[i] == op_node) {
        out_node->inputs[i] = recover_padding_op_node;
        recover_padding_op_node->outputs.push_back(out_node);
      }
    }

    // rename
    op_node->Op()->RenameOutput(out_node->Name(), recover_padding_input_name);
  };

  GraphPatternDetector gpd1;
  patterns::SkipLayernorm skip_layernorm(gpd1.mutable_pattern(),
                                         "remove_padding_recover_padding_pass");
  skip_layernorm();

  auto handler1 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "skip_layernorm";

    GET_IR_NODE_FROM_SUBGRAPH(skip_layernorm_x, skip_layernorm_x,
                              skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(skip_layernorm_y, skip_layernorm_y,
                              skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(skip_layernorm_op, skip_layernorm_op,
                              skip_layernorm);
    GET_IR_NODE_FROM_SUBGRAPH(skip_layernorm_out, skip_layernorm_out,
                              skip_layernorm);

    insert_remove_padding_op(skip_layernorm_x, skip_layernorm_op);
    insert_remove_padding_op(skip_layernorm_y, skip_layernorm_op);
    insert_recover_padding_op(skip_layernorm_op, skip_layernorm_out);

    found_subgraph_count++;
  };
  gpd1(graph, handler1);

  GraphPatternDetector gpd2;
  patterns::MultiheadMatmul multihead_matmul(
      gpd2.mutable_pattern(), "remove_padding_recover_padding_pass");
  multihead_matmul();

  auto handler2 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: "
               "multihead_matmul";

    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_input, multihead_matmul_input,
                              multihead_matmul);
    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_op, multihead_matmul_op,
                              multihead_matmul);
    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_out, multihead_matmul_out,
                              multihead_matmul);

    insert_remove_padding_op(multihead_matmul_input, multihead_matmul_op);
    insert_recover_padding_op(multihead_matmul_op, multihead_matmul_out);

    found_subgraph_count++;
  };
  gpd2(graph, handler2);

  GraphPatternDetector gpd3;
  patterns::Fc fc(gpd3.mutable_pattern(),
                  "remove_padding_recover_padding_pass");
  fc();

  auto handler3 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3) << "remove_padding_recover_padding_pass for transformer: fc";

    GET_IR_NODE_FROM_SUBGRAPH(fc_input, fc_input, fc);
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc_op, fc);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, fc);

    insert_remove_padding_op(fc_input, fc_op);
    insert_recover_padding_op(fc_op, fc_out);

    found_subgraph_count++;
  };
  gpd3(graph, handler3);

  GraphPatternDetector gpd4;
  patterns::Activation activation(gpd4.mutable_pattern(),
                                  "remove_padding_recover_padding_pass");
  activation();

  auto handler4 = [&](const GraphPatternDetector::subgraph_t& subgraph,
                      Graph* graph) {
    VLOG(3)
        << "remove_padding_recover_padding_pass for transformer: activation";

    GET_IR_NODE_FROM_SUBGRAPH(activation_input, activation_input, activation);
    GET_IR_NODE_FROM_SUBGRAPH(activation_op, activation_op, activation);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, activation);

    insert_remove_padding_op(activation_input, activation_op);
    insert_recover_padding_op(activation_op, activation_out);

    found_subgraph_count++;
  };
  gpd4(graph, handler4);

  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_padding_recover_padding_pass,
              paddle::framework::ir::RemovePaddingRecoverPaddingPass);
