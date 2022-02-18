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
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
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

// Checking whether a reorder from FP32 to BF16 should be added before the input
// to the operator
bool IsPermittedInputName(const std::string& input_name) {
  // Only the inputs listed in \"permitted_names\" requires quanitization before
  // the bfloat16 operator. Other inputs, such as Filter and Bias are reordered
  // in the kernel.
  const std::vector<std::string> permitted_names = {"X", "Y", "Input",
                                                    "ResidualData"};
  return (std::find(permitted_names.begin(), permitted_names.end(),
                    input_name) != permitted_names.end());
}

// Checking whether a reorder from BF16 to FP32 should be added after the output
// to the operator
bool IsPermittedOutputName(const std::string& output_name) {
  // XShape is output in transpose2 and reshape2 operators used to store the
  // shape and lod of X. So this output do not need dequantize before.
  return (output_name != "XShape");
}

void AddQuantize(Graph* g, ir::Node* op, ir::Node* op_in,
                 int& quantize_counter) {
  std::vector<std::string> input_names;

  // Find the name of the input linking op to op_in
  for (auto name : op->Op()->InputNames())
    for (auto input_name : op->Op()->Input(name))
      if (input_name == op_in->Name() && IsPermittedInputName(name))
        input_names.push_back(name);

  if (input_names.empty()) return;

  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({op_in->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", 1.f);
  q_desc.SetAttr("Shift", 0.0f);
  q_desc.SetAttr("bfloat16", true);
  q_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                      ? op->Op()->GetAttr("data_layout")
                                      : std::string("NCHW"));
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  for (auto name = input_names.begin(); name < input_names.end(); name++)
    op->Op()->SetInput(*name,
                       std::vector<std::string>({quantize_out_node->Name()}));

  UnlinkNodes(op_in, op);
  IR_NODE_LINK_TO(op_in, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
  quantize_counter++;
}

void AddQuantizes(Graph* g, ir::Node* op, int& quantize_counter) {
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
    q_desc.SetAttr("Shift", 0.0f);
    q_desc.SetAttr("bfloat16", true);
    q_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                        ? op->Op()->GetAttr("data_layout")
                                        : std::string("NCHW"));
    auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

    UnlinkNodes(inputs[i], op);
    IR_NODE_LINK_TO(inputs[i], quantize_op);
    IR_NODE_LINK_TO(quantize_op, quantize_out_nodes[i]);
    IR_NODE_LINK_TO(quantize_out_nodes[i], op);
    quantize_counter++;
  }

  op->Op()->SetInput("X", quantize_out_node_names);
}

// Operators like Concat and Sum have a single input name X, which actually
// consists of multiple inputs. Such operators require a different way to find
// pattern and add quantize ops.
void AddReoderBeforeDuplicatedInputs(ir::Graph* graph, int& quantize_counter) {
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

// Adding quantize ops before all operators except Concat and Sum, which have
// already been handled in AddReoderBeforeDuplicatedInputs
void AddReoderBeforeSingleInputs(ir::Graph* graph, int& quantize_counter) {
  GraphPatternDetector gpd;
  patterns::FirstBfloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                          "first_bfloat16_ops"};
  bfloat16_ops();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op_in, op_in, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);
    if (op->Op()->Type() != "sum" && op->Op()->Type() != "concat") {
      AddQuantize(g, op, op_in, quantize_counter);
    }
  };
  gpd(graph, handler);
}

void CPUBFloat16Pass::SetInputDataType(ir::Graph* graph) const {
  int quantize_counter = 0;
  AddReoderBeforeDuplicatedInputs(graph, quantize_counter);
  AddReoderBeforeSingleInputs(graph, quantize_counter);
  PrettyLogDetail("---    added %d quantize ops before bfloat16 op",
                  quantize_counter);
}

void AddDequantize(Graph* g, ir::Node* op, ir::Node* op_out,
                   int& dequantize_counter) {
  if (op->Op()->Type() == "prior_box") return;

  // Find the name of the output linking op to op_out
  std::vector<std::string> output_names;
  for (auto name : op->Op()->OutputNames())
    for (auto output_name : op->Op()->Output(name))
      if (output_name == op_out->Name() && IsPermittedOutputName(name))
        output_names.push_back(name);

  if (output_names.empty()) return;

  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({op_out->Name()}));
  deq_desc.SetAttr("Scale", 1.0f);
  deq_desc.SetAttr("Shift", 0.0f);
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  for (auto name = output_names.begin(); name < output_names.end(); name++)
    op->Op()->SetOutput(*name,
                        std::vector<std::string>({dequantize_in_node->Name()}));

  UnlinkNodes(op, op_out);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, op_out);

  dequantize_counter++;
}

void AddDequantizes(Graph* g, ir::Node* op, int& dequantize_counter) {
  auto outputs = op->outputs;
  PADDLE_ENFORCE_GE(outputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s outputs(%d) must be equal or greater than 1.",
                        op->Name(), outputs.size()));
  PADDLE_ENFORCE_EQ(op->inputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s inputs(%d) must be equal to 1.", op->Name(),
                        op->inputs.size()));

  OpDesc deq_desc;
  deq_desc.SetType("dequantize");

  std::vector<Node*> dequantize_in_nodes(outputs.size());
  std::vector<std::string> dequantize_in_node_names(outputs.size());

  for (size_t i = 0; i < outputs.size(); i++) {
    VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
    dequantize_in_nodes[i] = g->CreateVarNode(&dequantize_in_desc);
    dequantize_in_node_names[i] = dequantize_in_nodes[i]->Name();

    deq_desc.SetInput("Input",
                      std::vector<std::string>({dequantize_in_node_names[i]}));
    deq_desc.SetOutput("Output",
                       std::vector<std::string>({outputs[i]->Name()}));

    deq_desc.SetAttr("Scale", 1.f);
    deq_desc.SetAttr("Shift", 0.0f);
    deq_desc.SetAttr("bfloat16", true);
    deq_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                          ? op->Op()->GetAttr("data_layout")
                                          : std::string("NCHW"));
    auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

    UnlinkNodes(op, outputs[i]);
    IR_NODE_LINK_TO(op, dequantize_in_nodes[i]);
    IR_NODE_LINK_TO(dequantize_in_nodes[i], dequantize_op);
    IR_NODE_LINK_TO(dequantize_op, outputs[i]);

    dequantize_counter++;
  }

  op->Op()->SetOutput("Out", dequantize_in_node_names);
}

// Operators like split have a single output name Out, which actually
// consists of multiple outputs. Such operators require a different way to find
// pattern and add dequantize ops.
void AddReoderAfterDuplicatedOutputs(ir::Graph* graph,
                                     int& dequantize_counter) {
  GraphPatternDetector gpd;
  patterns::DuplicatedOutputs duplicated_outputs{gpd.mutable_pattern(),
                                                 "duplicated_outputs"};
  duplicated_outputs();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, duplicated_outputs);
    AddDequantizes(g, op, dequantize_counter);
  };
  gpd(graph, handler);
}

// Adding dequantize ops after all operators except split, which has
// already been handled in AddReoderAfterDuplicatedOutputs
void AddReoderAfterSingleOutputs(ir::Graph* graph, int& dequantize_counter) {
  GraphPatternDetector gpd;
  patterns::LastBfloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                         "last_bfloat16_ops"};
  bfloat16_ops();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op_out, op_out, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);
    if (op->Op()->Type() != "split") {
      AddDequantize(g, op, op_out, dequantize_counter);
    }
  };
  gpd(graph, handler);
}

void CPUBFloat16Pass::SetOutputDataType(ir::Graph* graph) const {
  int dequantize_counter = 0;
  AddReoderAfterDuplicatedOutputs(graph, dequantize_counter);
  AddReoderAfterSingleOutputs(graph, dequantize_counter);
  PrettyLogDetail("---    added %d dequantize ops after bfloat16 op",
                  dequantize_counter);
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
