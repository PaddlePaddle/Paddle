// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.3 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.3
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/add_support_int8_pass.h"

namespace paddle {
namespace framework {
namespace ir {

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

    // If inputs'tensors have the inputs_scale, then save it's index in
    // input_quant_tensor_index
    // OP'Attr hasn't std::vector<std::pair< >>. To do: Support multi-tensor
    // scale for one input
    /*
    for (size_t i = 0; i< quant_op->Op()->InputNames().size() ; i++){
      for (size_t j =0; j<
    quant_op->Op()->Input(quant_op->Op()->InputNames()[i]).size();j++){
        if(quant_op->Op()->HasAttr("Input_scale_"+quant_op->Op()->Input(quant_op->Op()->InputNames()[i])[j])){
          inscale_flag = true;
          input_quant_tensor_index.push_back(std::make_pair(i,j));
          inputs_scale.push_back(BOOST_GET_CONST(float,
    quant_op->Op()->GetAttr("Input_scale_"+quant_op->Op()->Input(quant_op->Op()->InputNames()[i])[j])));
        }
      }
    }
    */
    for (size_t i = 0; i < quant_op->Op()->InputNames().size(); i++) {
      if (quant_op->Op()->Input(quant_op->Op()->InputNames()[i]).size() > 0 &&
          quant_op->Op()->HasAttr(
              "Input_scale_" +
              quant_op->Op()->Input(quant_op->Op()->InputNames()[i])[0])) {
        inscale_flag = true;
        quant_op->Op()->SetAttr(
            quant_op->Op()->InputNames()[i],
            quant_op->Op()->GetAttr(
                "Input_scale_" +
                quant_op->Op()->Input(quant_op->Op()->InputNames()[i])[0]));
      }
    }

    // If outputs'tensors have the outputs_scale, then save it's index in
    // output_quant_tensor_index
    // OP'Attr hasn't std::vector<std::pair< >>. To do: Support multi-tensor
    // scale for one output
    /*
    for(auto out_node : quant_op->outputs){
      for (auto out_op_node : out_node->outputs){
        for (auto name : out_op_node->Op()->InputNames()){
          for (auto input_name : out_op_node->Op()->Input(name)){
            if(out_op_node->Op()->HasAttr("Input_scale_"+input_name)){
              for (size_t i = 0; i< quant_op->Op()->OutputNames().size() ; i++){
                for (size_t j =0; j<
    quant_op->Op()->Output(quant_op->Op()->OutputNames()[i]).size();j++){
                  if(input_name ==
    quant_op->Op()->Output(quant_op->Op()->OutputNames()[i])[j]){
                    outscale_flag = true;
                    output_quant_tensor_index.push_back(std::make_pair(i,j));
                    outputs_scale.push_back(BOOST_GET_CONST(float,
    out_op_node->Op()->GetAttr("Input_scale_"+input_name)));
                  }
                }
              }
            }
          }
        }
      }
    }
    */
    for (auto out_node : quant_op->outputs) {
      for (auto out_op_node : out_node->outputs) {
        for (auto name : out_op_node->Op()->InputNames()) {
          for (auto input_name : out_op_node->Op()->Input(name)) {
            if (out_op_node->Op()->HasAttr("Input_scale_" + input_name)) {
              for (size_t i = 0; i < quant_op->Op()->OutputNames().size();
                   i++) {
                if (quant_op->Op()
                            ->Output(quant_op->Op()->OutputNames()[i])
                            .size() > 0 &&
                    input_name ==
                        quant_op->Op()->Output(
                            quant_op->Op()->OutputNames()[i])[0]) {
                  outscale_flag = true;
                  quant_op->Op()->SetAttr(
                      quant_op->Op()->OutputNames()[i],
                      out_op_node->Op()->GetAttr("Input_scale_" + input_name));
                }
              }
            }
          }
        }
      }
    }
    quant_op->Op()->SetAttr("support_int8", inscale_flag && outscale_flag);

    quant_op->Op()->Flush();
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_support_int8_pass, paddle::framework::ir::AddSupportInt8Pass);
