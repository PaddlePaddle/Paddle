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

#include "paddle/fluid/framework/ir/delete_quant_dequant_op_pass.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                         \
  GET_IR_NODE(any_op_out);                \
  GET_IR_NODE(quant_dequant_op_inscale);  \
  GET_IR_NODE(quant_dequant_op);          \
  GET_IR_NODE(quant_dequant_op_outscale); \
  GET_IR_NODE(quant_dequant_op_out);      \
  GET_IR_NODE(any_op2);

void DeleteQuantDequantOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_quantdequant_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  patterns::DeleteQuantDequantOpPattern pattern(gpd.mutable_pattern(),
                                                pattern_name);
  pattern();
  auto* scope = param_scope();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    IR_NODE_LINK_TO(any_op_out, any_op2);
    std::string any_op_out_name = any_op_out->Var()->Name();
    std::string quant_dequant_op_out_name = quant_dequant_op_out->Var()->Name();

    std::string input_scale_var_name =
        quant_dequant_op->Op()->Input("InScale").front();
    const LoDTensor& input_scale_tensor =
        scope->FindVar(input_scale_var_name)->Get<LoDTensor>();

    const float* input_scale_data = input_scale_tensor.data<float>();
    float input_scale = input_scale_data[0];
    auto* any_op2_desc = any_op2->Op();
    // auto input_args_names = any_op2_desc->InputArgumentNames();
    auto var_map = any_op2_desc->Inputs();
    std::string arg_name = "";
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(), name_m.second.end(),
                    quant_dequant_op_out_name) != name_m.second.end()) {
        arg_name = name_m.first;
      }
    }
    CHECK(arg_name.size() > 0) << "can not find the input "
                               << quant_dequant_op_out_name;
    any_op2_desc->SetAttr("enable_int8", true);
    any_op2_desc->SetAttr(arg_name + "_scale", input_scale);

    // modify the any_op2's inputs
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(), name_m.second.end(),
                    quant_dequant_op_out_name) != name_m.second.end()) {
        std::vector<std::string> new_inputs;
        for (auto& i_n : name_m.second) {
          if (i_n != quant_dequant_op_out_name) {
            new_inputs.push_back(i_n);
          }
        }
        new_inputs.push_back(any_op_out_name);
        any_op2_desc->SetInput(name_m.first, new_inputs);
        any_op2_desc->Flush();
      }
    }
    any_op2_desc->Flush();
    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph,
                         {quant_dequant_op, quant_dequant_op_out,
                          quant_dequant_op_inscale, quant_dequant_op_outscale});
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_quant_dequant_op_pass,
              paddle::framework::ir::DeleteQuantDequantOpPass);
