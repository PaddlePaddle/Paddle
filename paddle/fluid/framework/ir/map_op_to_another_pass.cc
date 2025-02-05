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

#include "paddle/fluid/framework/ir/map_op_to_another_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

void MapOp2AnotherPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("map_op_to_another_pass", graph);

  int found_count = 0;
  std::unordered_map<std::string, std::string> replaced_map{
      {"conv2d", "conv2d"},
      {"depthwise_conv2d", "conv2d"},
      {"flatten_contiguous_range", "reshape2"},
  };

  auto nodes = graph->Nodes();

  for (auto& node : nodes) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    std::string op_type = op_desc->Type();
    if (!replaced_map.count(op_type)) continue;
    if (op_type == "flatten_contiguous_range") {
      auto start_axis = PADDLE_GET_CONST(int, op_desc->GetAttr("start_axis"));
      auto stop_axis = PADDLE_GET_CONST(int, op_desc->GetAttr("stop_axis"));
      auto input_name = op_desc->Input("X")[0];
      auto* block = op_desc->Block();
      auto input_shape = block->FindVar(input_name)->GetShape();
      if (start_axis == 1 && stop_axis == 3 && input_shape.size() == 4 &&
          input_shape[2] == 1 && input_shape[3] == 1) {
        op_desc->SetType(replaced_map[op_type]);
        op_desc->SetAttr("shape", std::vector<int>{0, -1});
      } else if (start_axis == 2 && stop_axis == 3 && input_shape.size() == 4 &&
                 input_shape[2] == 1) {
        op_desc->SetType(replaced_map[op_type]);
        op_desc->SetAttr("shape", std::vector<int>{0, 0, -1});
      }
    } else if (op_type == "depthwise_conv2d") {
      auto groups = PADDLE_GET_CONST(int, op_desc->GetAttr("groups"));
      if (groups > 1) {
#if CUDNN_VERSION >= 8100
        op_desc->SetType(replaced_map[op_type]);
        op_desc->RemoveAttr("use_cudnn");
        op_desc->SetAttr("use_cudnn", true);
#endif
      }
    } else if (op_type == "conv2d") {
      op_desc->SetType(replaced_map[op_type]);
      op_desc->RemoveAttr("use_cudnn");
      op_desc->SetAttr("use_cudnn", true);
    }
    op_desc->Flush();
    ++found_count;
  }

  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(map_op_to_another_pass, paddle::framework::ir::MapOp2AnotherPass);
REGISTER_PASS_CAPABILITY(map_op_to_another_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("depthwise_conv2d", 1)
            .LE("conv2d", 1)
            .EQ("reshape2", 0)
            .EQ("flatten_contiguous_range", 0));
