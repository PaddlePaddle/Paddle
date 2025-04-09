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

#include "paddle/fluid/framework/ir/add_support_int8_pass.h"

namespace paddle::framework::ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES GET_IR_NODE(quant_op);

void AddSupportInt8Pass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "add_support_int8";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  patterns::AddSupportInt8 pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    bool inscale_flag = false;
    bool outscale_flag = false;
    auto* quanted_op_desc = quant_op->Op();
    // If inputs'tensors have the inputs_scale, then save it's index in
    // input_quant_tensor_index
    // OP'Attr hasn't std::vector<std::pair< >>. To do: Support multi-tensor
    // scale for one input
    for (size_t i = 0; i < quanted_op_desc->InputNames().size(); i++) {
      if (!quanted_op_desc->Input(quanted_op_desc->InputNames()[i]).empty() &&
          quanted_op_desc->HasAttr(
              "Input_scale_" +
              quanted_op_desc->Input(quanted_op_desc->InputNames()[i])[0])) {
        inscale_flag = true;
        quanted_op_desc->SetAttr(
            quanted_op_desc->InputNames()[i],
            quanted_op_desc->GetAttr(
                "Input_scale_" +
                quanted_op_desc->Input(quanted_op_desc->InputNames()[i])[0]));
      }
    }

    // If outputs'tensors have the outputs_scale, then save it's index in
    // output_quant_tensor_index
    // OP'Attr hasn't std::vector<std::pair< >>. To do: Support multi-tensor
    // scale for one output
    for (auto out_node : quant_op->outputs) {
      for (auto out_op_node : out_node->outputs) {
        for (auto const& name : out_op_node->Op()->InputNames()) {
          for (auto const& input_name : out_op_node->Op()->Input(name)) {
            if (out_op_node->Op()->HasAttr("Input_scale_" + input_name)) {
              for (size_t i = 0; i < quanted_op_desc->OutputNames().size();
                   i++) {
                if (!quanted_op_desc->Output(quanted_op_desc->OutputNames()[i])
                         .empty() &&
                    input_name == quanted_op_desc->Output(
                                      quanted_op_desc->OutputNames()[i])[0]) {
                  outscale_flag = true;
                  quanted_op_desc->SetAttr(
                      quanted_op_desc->OutputNames()[i],
                      out_op_node->Op()->GetAttr("Input_scale_" + input_name));
                }
              }
            }
          }
        }
      }
    }
    quanted_op_desc->SetAttr("support_int8", inscale_flag && outscale_flag);
    quanted_op_desc->Flush();
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(add_support_int8_pass, paddle::framework::ir::AddSupportInt8Pass);
