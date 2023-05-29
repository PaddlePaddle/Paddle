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

#include "paddle/fluid/framework/ir/remove_shape_tensor_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void RemoveShapeTensorPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("remove_shape_tensor_pass", graph);


  auto op_node_sorted = framework::ir::TopologyVarientSort(
      *graph, static_cast<framework::ir::SortKind>(0));
  for (auto *op_node : op_node_sorted) {

    auto op_desc = op_node->Op();

    if(op_desc->Type() == "slice") {
        // slice如果他有shape tensor输入，且为固定的数，那么
        if(op_desc->HasInput("EndsTensorList") && op_desc->Input("EndsTensorList").size() == 1L) {
            auto end_tensor_list_inputs = op_desc->Input("EndsTensorList");
            std::cout << end_tensor_list_inputs[0] << std::endl;
            op_desc->RemoveInput("EndsTensorList");
            op_desc->SetInput("EndsTensor", end_tensor_list_inputs);
        }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_shape_tensor_pass,
              paddle::framework::ir::RemoveShapeTensorPass);
REGISTER_PASS_CAPABILITY(remove_shape_tensor_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("scale", 0)
            .LE("c_identity", 1));
