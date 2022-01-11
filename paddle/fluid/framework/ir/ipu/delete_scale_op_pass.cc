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

#include "paddle/fluid/framework/ir/ipu/delete_scale_op_pass.h"

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"

namespace paddle {
namespace framework {
namespace ir {

// this pass is used to optimize scale operator whose scale = 1 and bias = 0.
// scale will not be optimized if it is the only one operator in the graph.
void DeleteScaleOpPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter DeleteScaleOpPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto nodes = ir::TopologySortOperations(*graph);

  // delete op
  for (auto node : nodes) {
    if (!node->Op()) {
      continue;
    }
    auto op = node->Op();
    if (op->Type() == "scale") {
      auto input_var_node = node->inputs[0];
      auto output_var_node = node->outputs[0];
      // only optimize scale *1 + 0
      auto scale = BOOST_GET_CONST(float, op->GetAttr("scale"));
      auto bias = BOOST_GET_CONST(float, op->GetAttr("bias"));
      if (scale != 1 || bias != 0) {
        return;
      }
      // only one op and it is scale , do not optimize
      if (input_var_node->inputs.size() == 0 &&
          output_var_node->outputs.size() == 0) {
        return;
      }
      VLOG(10) << "scale is to be optimized "
               << " scale: " << scale << " bias: " << bias;
      // build link in nodes
      ir::Node* next_op_node = nullptr;
      ir::Node* pre_op_node = nullptr;
      // scale is not the last one
      if (node->outputs[0]->outputs.size() > 0) {
        next_op_node = node->outputs[0]->outputs[0];
        input_var_node->outputs.push_back(next_op_node);
        next_op_node->inputs.push_back(input_var_node);
        platform::ipu::DisConnectNodes(output_var_node, node);
        platform::ipu::DisConnectNodes(input_var_node, node);
        auto var_map = next_op_node->Op()->Inputs();
        for (auto& name_m : var_map) {
          if (std::find(name_m.second.begin(), name_m.second.end(),
                        output_var_node->Name()) != name_m.second.end()) {
            std::vector<std::string> new_inputs;
            for (auto& i_n : name_m.second) {
              if (i_n != output_var_node->Name()) {
                new_inputs.push_back(i_n);
              }
            }
            new_inputs.push_back(input_var_node->Name());
            next_op_node->Op()->SetInput(name_m.first, new_inputs);
            next_op_node->Op()->Flush();
          }
        }
        GraphSafeRemoveNodes(graph, {node, output_var_node});
      } else {  // scale is not the first one
        pre_op_node = node->inputs[0]->inputs[0];
        output_var_node->inputs.push_back(pre_op_node);
        pre_op_node->outputs.push_back(output_var_node);
        platform::ipu::DisConnectNodes(input_var_node, node);
        platform::ipu::DisConnectNodes(output_var_node, node);

        auto var_map = pre_op_node->Op()->Inputs();
        std::vector<std::string> new_outputs;
        for (auto& name_m : var_map) {
          if (std::find(name_m.second.begin(), name_m.second.end(),
                        input_var_node->Name()) != name_m.second.end()) {
            for (auto& i_n : name_m.second) {
              if (i_n != input_var_node->Name()) {
                new_outputs.push_back(i_n);
              }
            }
            new_outputs.push_back(output_var_node->Name());
            pre_op_node->Op()->SetOutput(name_m.first, new_outputs);
            pre_op_node->Op()->Flush();
          }
        }
        GraphSafeRemoveNodes(graph, {node, input_var_node});
      }
    }
  }
  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave DeleteScaleOpPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_scale_op_pass, paddle::framework::ir::DeleteScaleOpPass);
