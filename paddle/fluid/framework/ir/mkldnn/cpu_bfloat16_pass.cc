/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_pass.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

void AddQuantize(Graph* g, ir::Node* op, ir::Node* op_in,
                 int* quantize_counter) {
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({op_in->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", 1.f);
  q_desc.SetAttr("bfloat16", true);
  q_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                      ? op->Op()->GetAttr("data_layout")
                                      : std::string("NCHW"));
  auto quantize_op = g->CreateOpNode(&q_desc);

  std::vector<std::string> input_names;
  for (auto name : op->Op()->InputNames()) {
    for (auto input_name : op->Op()->Input(name)) {
      if (input_name == op_in->Name()) input_names.push_back(name);
    }
  }

  PADDLE_ENFORCE_NE(
      input_names.empty(), true,
      platform::errors::NotFound(
          "Operator before operator should have input as op output"));

  for (auto name = input_names.begin(); name < input_names.end(); name++)
    op->Op()->SetInput(*name,
                       std::vector<std::string>({quantize_out_node->Name()}));

  UnlinkNodes(op_in, op);
  IR_NODE_LINK_TO(op_in, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
  (*quantize_counter)++;
}

void AddQuantizes(Graph* g, ir::Node* op, int* quantize_counter) {
  auto inputs = op->inputs;
  PADDLE_ENFORCE_GE(inputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s inputs(%d) must be equal or greater than 1.",
                        op->Name(), inputs.size()));
  PADDLE_ENFORCE_EQ(op->outputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s outputs(%d) must be equal to 1.", op->Name(),
                        op->outputs.size()));

  OpDesc q_desc;
  q_desc.SetType("quantize");

  std::vector<Node*> quantize_out_nodes(inputs.size());
  std::vector<std::string> quantize_out_node_names(inputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
    quantize_out_nodes[i] = g->CreateVarNode(&quantize_out_desc);
    quantize_out_node_names[i] = quantize_out_nodes[i]->Name();

    q_desc.SetInput("Input", std::vector<std::string>({inputs[i]->Name()}));
    q_desc.SetOutput("Output",
                     std::vector<std::string>({quantize_out_node_names[i]}));
    q_desc.SetAttr("Scale", 1.f);
    q_desc.SetAttr("bfloat16", true);
    q_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                        ? op->Op()->GetAttr("data_layout")
                                        : std::string("NCHW"));
    auto quantize_op = g->CreateOpNode(&q_desc);

    UnlinkNodes(inputs[i], op);
    IR_NODE_LINK_TO(inputs[i], quantize_op);
    IR_NODE_LINK_TO(quantize_op, quantize_out_nodes[i]);
    IR_NODE_LINK_TO(quantize_out_nodes[i], op);
    (*quantize_counter)++;
  }

  op->Op()->SetInput("X", quantize_out_node_names);
}

void AddReoderBeforeDuplicatedInputs(ir::Graph* graph, int* quantize_counter) {
  GraphPatternDetector gpd;
  patterns::DuplicatedInputs duplicated_inputs{gpd.mutable_pattern(),
                                               "duplicated_inputs"};
  duplicated_inputs();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, duplicated_inputs);
    AddQuantizes(g, op, quantize_counter);
  };
  gpd(graph, handler);
}

void RemoveUnnecessaryReorders(ir::Graph* graph, int* quantize_counter) {
  GraphPatternDetector gpd;
  patterns::UnnecessaryReorders unnecessary_reorders{gpd.mutable_pattern(),
                                                     "unnecessary_reorders"};
  unnecessary_reorders();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, unnecessary_reorders);
    GET_IR_NODE_FROM_SUBGRAPH(quant_in, quant_in, unnecessary_reorders);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, unnecessary_reorders);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, unnecessary_reorders);

    std::string op_output_name;
    for (auto name : prev_op->Op()->OutputNames())
      for (auto output_name : prev_op->Op()->Output(name))
        if (output_name == quant_in->Name()) op_output_name = name;

    PADDLE_ENFORCE_NE(
        op_output_name.empty(), true,
        platform::errors::NotFound(
            "Operator before operator should have input as op output"));

    prev_op->Op()->SetOutput(op_output_name,
                             std::vector<std::string>({quant_out->Name()}));

    IR_NODE_LINK_TO(prev_op, quant_out);
    GraphSafeRemoveNodes(graph, {quant_in, quant_op});
    (*quantize_counter)--;
  };
  gpd(graph, handler);
}

void AddReoderBeforeSingleInputs(ir::Graph* graph, int* quantize_counter) {
  GraphPatternDetector gpd;
  patterns::FirstBfloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                          "first_bfloat16_ops"};
  bfloat16_ops();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op_in, op_in, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);
    auto prev_op_type = prev_op->Op()->Type();
    if (op->Op()->Type() != "conv2d" && prev_op_type != "quantize" &&
        prev_op_type != "sum" && prev_op_type != "concat") {
      AddQuantize(g, op, op_in, quantize_counter);
    }
  };
  gpd(graph, handler);
}

void CPUBFloat16Pass::SetInputDataType(ir::Graph* graph) const {
  int quantize_counter = 0;
  AddReoderBeforeDuplicatedInputs(graph, &quantize_counter);
  RemoveUnnecessaryReorders(graph, &quantize_counter);
  AddReoderBeforeSingleInputs(graph, &quantize_counter);
  PrettyLogDetail("---    added %d quantize op before bfloat16 op",
                  quantize_counter);
}

void CPUBFloat16Pass::SetOutputDataType(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::LastBfloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                         "last_bfloat16_ops"};
  bfloat16_ops();
  int force_fp32_counter = 0, dequantize_counter = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op_out, op_out, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, bfloat16_ops);
    if ((op->Op()->HasAttr("force_fp32_output") ||
         op->Op()->HasProtoAttr("force_fp32_output")) &&
        !op->Op()->GetAttrIfExists<bool>("fuse_residual_connection")) {
      op->Op()->SetAttr("force_fp32_output", true);
      force_fp32_counter++;
    } else if (op->Op()->Type() != "prior_box") {
      VarDesc dequantize_out_desc(patterns::PDNodeName("dequantize", "out"));
      auto* dequantize_out_node = g->CreateVarNode(&dequantize_out_desc);

      OpDesc deq_desc;
      deq_desc.SetType("dequantize");
      deq_desc.SetInput("Input", std::vector<std::string>({op_out->Name()}));
      deq_desc.SetOutput(
          "Output", std::vector<std::string>({dequantize_out_node->Name()}));
      deq_desc.SetAttr("Scale", 1.0f);
      auto dequantize_op = g->CreateOpNode(&deq_desc);

      std::string next_op_input_name;
      for (auto name : next_op->Op()->InputNames()) {
        for (auto input_name : next_op->Op()->Input(name)) {
          if (input_name == op_out->Name()) next_op_input_name = name;
        }
      }

      PADDLE_ENFORCE_NE(
          next_op_input_name.empty(), true,
          platform::errors::NotFound(
              "Operator before operator should have input as op output"));

      next_op->Op()->SetInput(
          next_op_input_name,
          std::vector<std::string>({dequantize_out_node->Name()}));
      UnlinkNodes(op_out, next_op);
      IR_NODE_LINK_TO(op_out, dequantize_op);
      IR_NODE_LINK_TO(dequantize_op, dequantize_out_node);
      IR_NODE_LINK_TO(dequantize_out_node, next_op);
      dequantize_counter++;
    }
  };
  gpd(graph, handler);
  PrettyLogDetail("---    added %d dequantize op and used %d force_fp32_output",
                  dequantize_counter, force_fp32_counter);
}

void CPUBFloat16Pass::ApplyImpl(ir::Graph* graph) const {
  SetInputDataType(graph);
  SetOutputDataType(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_pass, paddle::framework::ir::CPUBFloat16Pass);

REGISTER_PASS_CAPABILITY(cpu_bfloat16_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "quantize", 1));
