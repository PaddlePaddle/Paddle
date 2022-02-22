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

#include "paddle/fluid/framework/ir/delete_dropout_op_pass.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                  \
  GET_IR_NODE(any_op_out);         \
  GET_IR_NODE(dropout_op);         \
  GET_IR_NODE(dropout_op_out);     \
  GET_IR_NODE(dropout_op_outmask); \
  GET_IR_NODE(any_op2);

void DeleteDropoutOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_dropout_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  patterns::DeleteDropoutOpPattern pattern(gpd.mutable_pattern(), pattern_name);
  pattern();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    IR_NODE_LINK_TO(any_op_out, any_op2);
    std::string any_op_out_name = any_op_out->Var()->Name();
    std::string dropout_op_out_name = dropout_op_out->Var()->Name();

    auto* any_op2_desc = any_op2->Op();
    auto var_map = any_op2_desc->Inputs();
    std::string arg_name = "";
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(), name_m.second.end(),
                    dropout_op_out_name) != name_m.second.end()) {
        arg_name = name_m.first;
      }
    }
    if (arg_name.size() == 0) {
      LOG(INFO) << "Delete dropout op pass: can not find the input "
                << dropout_op_out_name;
      return;
    }

    // modify the any_op2's inputs
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(), name_m.second.end(),
                    dropout_op_out_name) != name_m.second.end()) {
        std::vector<std::string> new_inputs;
        for (auto& i_n : name_m.second) {
          if (i_n != dropout_op_out_name) {
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
                         {dropout_op, dropout_op_out, dropout_op_outmask});
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_dropout_op_pass,
              paddle::framework::ir::DeleteDropoutOpPass);
