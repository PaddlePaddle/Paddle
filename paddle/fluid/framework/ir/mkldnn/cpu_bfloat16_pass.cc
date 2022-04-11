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

void AddReoderAfterDuplicatedOutputs(ir::Graph* graph, int& dequantize_counter);
void AddReoderAfterSingleOutputs(ir::Graph* graph, int& dequantize_counter);

void AddQuantizes(Graph* g, ir::Node* op, int& quantize_counter);
void AddDequantizes(Graph* g, ir::Node* op, int& dequantize_counter);
void AddDequantize(Graph* g, ir::Node* op, ir::Node* op_out, int& dequantize_counter);

bool IsAlreadyLinked(const std::vector<std::string>& node_names, std::string node_name);
ir::Node* create_quantize_op(const std::string& input_name, const std::string& output_name, Graph* g, ir::Node* op);
void UnlinkNodes(ir::Node* a, ir::Node* b);
bool IsNotPermittedInputName(const std::string& input_name);
bool IsPermittedOutputName(const std::string& output_name);

void CPUBFloat16Pass::ApplyImpl(ir::Graph* graph) const {
  SetInputDataType(graph);
  SetOutputDataType(graph);
}

void CPUBFloat16Pass::SetInputDataType(ir::Graph* graph) const {
  int quantize_counter = 0;

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

  PrettyLogDetail("---    added %d quantize ops before bfloat16 op",
                  quantize_counter);
}

void CPUBFloat16Pass::SetOutputDataType(ir::Graph* graph) const {
  int dequantize_counter = 0;
  AddReoderAfterDuplicatedOutputs(graph, dequantize_counter);
  AddReoderAfterSingleOutputs(graph, dequantize_counter);
  PrettyLogDetail("---    added %d dequantize ops after bfloat16 op",
                  dequantize_counter);
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

void AddQuantizes(Graph* g, ir::Node* op, int& quantize_counter) {
  auto inputs = op->inputs;
  PADDLE_ENFORCE_GE(inputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s inputs(%d) must be equal or greater than 1.",
                        op->Name(), inputs.size()));

  std::map<std::string, ir::Node*> inputs_map;
  for(auto input: inputs)
    inputs_map[input->Name()] = input;

  std::vector<std::string> linked_inputs;

  auto op_inputs = op->Op()->Inputs();
  for(auto logical_input : op_inputs) {
    std::vector<std::string> quantize_output_names;
    quantize_output_names.reserve(inputs.size());

    auto logical_input_name = logical_input.first;
    if(IsNotPermittedInputName(logical_input_name))
      continue;

    auto physical_inputs_names = logical_input.second;
    for(auto physical_input_name: physical_inputs_names) {
      if(IsAlreadyLinked(linked_inputs, physical_input_name))
        continue;

      VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
      auto quantize_out_node = g->CreateVarNode(&quantize_out_desc);
      auto output_name = quantize_out_node->Name();
      quantize_output_names.emplace_back(output_name);

      auto quantize_op = create_quantize_op(physical_input_name, output_name, g, op);

      auto physical_input_node = inputs_map[physical_input_name];
      UnlinkNodes(physical_input_node, op);
      IR_NODE_LINK_TO(physical_input_node, quantize_op);
      IR_NODE_LINK_TO(quantize_op, quantize_out_node);
      IR_NODE_LINK_TO(quantize_out_node, op);
      quantize_counter++;
      linked_inputs.push_back(physical_input_name);
    }

    op->Op()->SetInput(logical_input_name, quantize_output_names);
  }
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

bool IsAlreadyLinked(const std::vector<std::string>& node_names, std::string node_name)
{
  return std::find(node_names.begin(), node_names.end(), node_name) != node_names.end();
}

ir::Node* create_quantize_op(const std::string& input_name, const std::string& output_name, Graph* g, ir::Node* op) {
  OpDesc q_desc;
  q_desc.SetType("quantize");

  q_desc.SetInput("Input", std::vector<std::string>({input_name}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({output_name}));
  q_desc.SetAttr("Scale", 1.f);
  q_desc.SetAttr("Shift", 0.0f);
  q_desc.SetAttr("bfloat16", true);
  q_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                      ? op->Op()->GetAttr("data_layout")
                                      : std::string("NCHW"));
  return g->CreateOpNode(&q_desc);  // OpDesc will be copied.
}

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

// Checking whether a reorder from FP32 to BF16 should be added before the input
// to the operator
bool IsNotPermittedInputName(const std::string& input_name) {
  // Only the inputs listed in \"permitted_names\" requires quanitization before
  // the bfloat16 operator. Other inputs, such as Filter and Bias are reordered
  // in the kernel.
  const std::vector<std::string> permitted_names = {"X", "Y", "Input",
                                                    "ResidualData"};

  return std::none_of(permitted_names.begin(),
                      permitted_names.end(),
                      [&input_name](std::string name) {
                        return name == input_name;
                      }
                     );
}

// Checking whether a reorder from BF16 to FP32 should be added after the output
// to the operator
bool IsPermittedOutputName(const std::string& output_name) {
  // XShape is output in transpose2 and reshape2 operators used to store the
  // shape and lod of X. So this output do not need dequantize before.
  return (output_name != "XShape");
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_pass, paddle::framework::ir::CPUBFloat16Pass);

REGISTER_PASS_CAPABILITY(cpu_bfloat16_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "quantize", 1));
