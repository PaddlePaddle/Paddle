// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/inplace_op_var_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void InplaceOpVarPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("inplace_op_var", graph);
  int found_subgraph_count = 0;
  auto nodes = graph->Nodes();

  // flatten_contiguous_range op map to reshape.
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "flatten_contiguous_range") {
      auto* op_node = node->Op();
      auto start_axis = PADDLE_GET_CONST(int, op_node->GetAttr("start_axis"));
      auto stop_axis = PADDLE_GET_CONST(int, op_node->GetAttr("stop_axis"));
      auto input_name = op_node->Input("X")[0];
      auto* block = op_node->Block();
      auto input_shape = block->FindVar(input_name)->GetShape();
      if (start_axis == 1 && stop_axis == 3 && input_shape.size() == 4 &&
          input_shape[2] == 1 && input_shape[3] == 1) {
        op_node->SetType("reshape2");
        op_node->SetAttr("shape", std::vector<int>{0, -1});
        op_node->Flush();
      }
    }
  }

  // inplace all reshape op.
  for (auto* node : nodes) {
    if (node->IsOp()) {
      auto* op_node = node->Op();
      if (op_node->Type() == "reshape2" && node->inputs.size() == 1) {
        ++found_subgraph_count;

        auto input_name = op_node->Input("X")[0];
        auto output_name = op_node->Output("Out")[0];

        for (auto* out_var : node->outputs) {
          if (out_var->Name() == output_name) {
            out_var->RenameVar(input_name);
            for (auto* next_op : out_var->outputs) {
              next_op->Op()->RenameInput(output_name, input_name);
              next_op->Op()->Flush();
            }
          }
        }

        op_node->RenameOutput(output_name, input_name);
        op_node->Flush();
      }
    }
  }
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_op_var_pass, paddle::framework::ir::InplaceOpVarPass);
REGISTER_PASS_CAPABILITY(inplace_op_var_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "reshape2", 0));
