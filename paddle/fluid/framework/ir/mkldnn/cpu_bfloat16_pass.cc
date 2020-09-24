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

void CPUBFloat16Pass::SetInputDataType(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::FirstBfloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                          "first_bfloat16_ops"};
  bfloat16_ops();
  int quantize_counter = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op_in, op_in, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);

    if (op->Op()->Type() != "conv2d" && prev_op->Op()->Type() != "quantize") {
      VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
      auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

      // create a quantize op node
      OpDesc q_desc;
      q_desc.SetType("quantize");
      q_desc.SetInput("Input", std::vector<std::string>({op_in->Name()}));
      q_desc.SetOutput("Output",
                       std::vector<std::string>({quantize_out_node->Name()}));
      q_desc.SetAttr("Scale", 1.f);
      q_desc.SetAttr("bfloat16", true);
      q_desc.SetAttr("output_format", Has("data_layout")
                                          ? Get<std::string>("data_layout")
                                          : "NCHW");
      auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

      std::string op_input_name;
      for (auto name : op->Op()->InputNames()) {
        for (auto input_name : op->Op()->Input(name)) {
          if (input_name == op_in->Name()) op_input_name = name;
        }
      }

      PADDLE_ENFORCE_NE(
          op_input_name.empty(), true,
          platform::errors::NotFound(
              "Operator before operator should have input as op output"));

      op->Op()->SetInput(op_input_name,
                         std::vector<std::string>({quantize_out_node->Name()}));

      UnlinkNodes(op_in, op);
      IR_NODE_LINK_TO(op_in, quantize_op);
      IR_NODE_LINK_TO(quantize_op, quantize_out_node);
      IR_NODE_LINK_TO(quantize_out_node, op);
      quantize_counter++;
    }
  };
  gpd(graph, handler);
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
      // Create dequantize input variable
      VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
      auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

      // create a dequantize op node for output.
      OpDesc deq_desc;
      deq_desc.SetType("dequantize");
      deq_desc.SetInput("Input",
                        std::vector<std::string>({dequantize_in_node->Name()}));
      deq_desc.SetOutput("Output", std::vector<std::string>({op_out->Name()}));
      deq_desc.SetAttr("Scale", 1.0f);
      auto dequantize_op = g->CreateOpNode(&deq_desc);

      std::string op_output_name;
      for (auto name : op->Op()->OutputNames()) {
        for (auto output_name : op->Op()->Output(name)) {
          if (output_name == op_out->Name()) op_output_name = name;
        }
      }

      PADDLE_ENFORCE_NE(
          op_output_name.empty(), true,
          platform::errors::NotFound(
              "Operator after operator should have input as op output"));

      op->Op()->SetOutput(op_output_name, std::vector<std::string>(
                                              {dequantize_in_node->Name()}));

      UnlinkNodes(op, op_out);
      IR_NODE_LINK_TO(op, dequantize_in_node);
      IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
      IR_NODE_LINK_TO(dequantize_op, op_out);
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
