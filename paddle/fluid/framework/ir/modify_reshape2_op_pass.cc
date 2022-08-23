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

#include "paddle/fluid/framework/ir/modify_reshape2_op_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
namespace ir {

void ModifyReshape2OpPass::ApplyImpl(ir::Graph* graph) const {
  return;
  FusePassBase::Init("modify_reshape2_op_pass", graph);
  GraphPatternDetector detector;
  auto reshape2_op = detector.mutable_pattern()
                         ->NewNode("reshape2_op")
                         ->assert_is_op("reshape2")
                         ->assert_more([&](Node* node) {
                           int op_input_size = node->inputs.size();
                           return op_input_size >= 2;
                         });
  // auto *scope = param_scope();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    Node* reshape2_op_node = subgraph.at(reshape2_op);
    auto shape_tensor_names = reshape2_op_node->Op()->Input("ShapeTensor");
    std::vector<paddle::framework::ir::Node*> not_remove_nodes;
    auto shape = PADDLE_GET_CONST(std::vector<int>, reshape2_op_node->Op()->GetAttr("shape"));
    int shape_max = *std::max_element(shape.begin(), shape.end());
    // if all ShapeTensor is undetermined, skip it
    if (shape_max < 0) {
      return;
    }

    for (auto input : reshape2_op_node->inputs) {
      if (std::find(shape_tensor_names.begin(), shape_tensor_names.end(),
                    input->Name()) != shape_tensor_names.end()) {
        std::vector<paddle::framework::ir::Node*> remain_nodes;
        for (auto i : input->outputs) {
          if (i != reshape2_op_node) {
            remain_nodes.push_back(i);
          }
        }
        input->outputs = remain_nodes;

      } else {
        not_remove_nodes.push_back(input);
      }
    }
    std::cout <<"shape_max" << shape_max << not_remove_nodes[0]->Name() << std::endl;
    reshape2_op_node->inputs = not_remove_nodes;
    reshape2_op_node->Op()->RemoveInput("ShapeTensor");
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] < 0) shape[i] = 0;
    }
    if (shape.size() == 1UL) shape[0] = -1;
    reshape2_op_node->Op()->SetAttr("shape", shape);

    // if (reshape2_op_node->Op()->Input("X")[0] != "fill_constant_113.tmp_0") {

    //   reshape2_op_node->Op()->RemoveInput("ShapeTensor");
    //   auto shape = PADDLE_GET_CONST(std::vector<int>,
    //                                 reshape2_op_node->Op()->GetAttr("shape"));
    //   for (size_t i = 0; i < shape.size(); i++) {
    //     if (shape[i] < 0) shape[i] = 0;
    //   }
    //   if (shape.size() == 1UL) shape[0] = -1;
    //   reshape2_op_node->Op()->SetAttr("shape", shape);
    // }

  };

if(1)  detector(graph, handler);

  while (true) {
    std::unordered_set<const paddle::framework::ir::Node*> remove_nodes;
    auto op_node_sorted = framework::ir::TopologyVarientSort(
        *graph, static_cast<framework::ir::SortKind>(0));
    for (auto* op_node : op_node_sorted) {
      if (!op_node->IsOp()) continue;
      if (op_node->Name() == "fetch") continue;
      bool all_output_useless = true;
      for (auto i : op_node->outputs) {
        if (i->outputs.size() >= 1) {
          all_output_useless = false;
        }
      }
      if (all_output_useless) {
        remove_nodes.emplace(op_node);
        for (auto i : op_node->outputs) {
          remove_nodes.emplace(i);
        }
      }
    }
    if (remove_nodes.empty()) break;
    GraphSafeRemoveNodes(graph, remove_nodes);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(modify_reshape2_op_pass,
              paddle::framework::ir::ModifyReshape2OpPass);
