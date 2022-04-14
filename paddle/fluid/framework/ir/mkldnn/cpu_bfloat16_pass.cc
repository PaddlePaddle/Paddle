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

void AddQuantizes(Graph* g, ir::Node* op, int& quantize_counter);
void AddDequantizes(Graph* g, ir::Node* op, int& dequantize_counter);

bool IsAlreadyLinked(const std::vector<std::string>& node_names, std::string node_name);
ir::Node* create_quant_op(const std::string& op_type, const std::string& input_name, const std::string& output_name, Graph* g, ir::Node* op);
void UnlinkNodes(ir::Node* a, ir::Node* b);
bool IsNotPermittedInputName(const std::string& input_name);
bool IsNotPermittedOutputName(const std::string& output_name);

void CPUBFloat16Pass::ApplyImpl(ir::Graph* graph) const {
  int quantize_counter = 0;
  int dequantize_counter = 0;

  GraphPatternDetector gpd;
  patterns::Bloat16Ops Bloat16Ops{gpd.mutable_pattern(), "Bloat16Ops"};
  Bloat16Ops();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, Bloat16Ops);
    AddQuantizes(g, op, quantize_counter);
    AddDequantizes(g, op, dequantize_counter);
  };
  gpd(graph, handler);

  PrettyLogDetail("---    added %d quantize ops before bfloat16 op",
                  quantize_counter);
  PrettyLogDetail("---    added %d dequantize ops after bfloat16 op",
                  dequantize_counter);
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

      auto quantize_op = create_quant_op("quantize", physical_input_name, output_name, g, op);

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

  std::map<std::string, ir::Node*> outputs_map;
  for(auto output: outputs)
    outputs_map[output->Name()] = output;

  std::vector<std::string> linked_outputs;

  auto op_outputs = op->Op()->Outputs();
  for(auto logical_output : op_outputs) {
    std::vector<std::string> dequantize_input_names;
    dequantize_input_names.reserve(outputs.size());

    auto logical_output_name = logical_output.first;
    if(IsNotPermittedOutputName(logical_output_name))
      continue;

    auto physical_outputs_names = logical_output.second;
    for(auto physical_output_name: physical_outputs_names) {
      if(IsAlreadyLinked(linked_outputs, physical_output_name))
        continue;

      VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
      auto dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);
      auto input_name = dequantize_in_node->Name();
      dequantize_input_names.emplace_back(input_name);

      auto dequantize_op = create_quant_op("dequantize", input_name, physical_output_name, g, op);

      auto physical_output_node = outputs_map[physical_output_name];
      UnlinkNodes(op, physical_output_node);
      IR_NODE_LINK_TO(dequantize_op, physical_output_node);
      IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
      IR_NODE_LINK_TO(op, dequantize_in_node);
      dequantize_counter++;
      linked_outputs.push_back(physical_output_name);
    }

    op->Op()->SetOutput(logical_output_name, dequantize_input_names);
  }
}

bool IsAlreadyLinked(const std::vector<std::string>& node_names, std::string node_name)
{
  return std::find(node_names.begin(), node_names.end(), node_name) != node_names.end();
}

ir::Node* create_quant_op(const std::string& op_type, const std::string& input_name, const std::string& output_name, Graph* g, ir::Node* op) {
  OpDesc op_desc;
  op_desc.SetType(op_type);

  op_desc.SetInput("Input", std::vector<std::string>({input_name}));
  op_desc.SetOutput("Output",
                   std::vector<std::string>({output_name}));
  op_desc.SetAttr("Scale", 1.f);
  op_desc.SetAttr("Shift", 0.0f);
  op_desc.SetAttr("bfloat16", true);
  op_desc.SetAttr("output_format", op->Op()->HasAttr("data_layout")
                                       ? op->Op()->GetAttr("data_layout")
                                       : std::string("NCHW"));
  return g->CreateOpNode(&op_desc);  // OpDesc will be copied.
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
bool IsNotPermittedOutputName(const std::string& output_name) {
  // XShape is output in transpose2 and reshape2 operators used to store the
  // shape and lod of X. So this output do not need dequantize before.
  return (output_name == "XShape");
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_pass, paddle::framework::ir::CPUBFloat16Pass);

REGISTER_PASS_CAPABILITY(cpu_bfloat16_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "quantize", 1));
