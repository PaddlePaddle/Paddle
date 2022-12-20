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

#include "paddle/fluid/framework/ir/map_depthwise_conv_to_conv_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MapDepthwiseConv2ConvPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("map_depthwise_conv_to_conv_pass", graph);

  int found_count = 0;
  std::unordered_map<std::string, std::string> replaced_map{
      {"depthwise_conv2d", "conv2d"},
  };

  auto nodes = graph->Nodes();

  for (auto& node : nodes) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    std::string op_type = op_desc->Type();
    if (!replaced_map.count(op_type)) continue;
    op_desc->SetType(replaced_map[op_type]);
    op_desc->SetAttr("use_cudnn", true);
    op_desc->Flush();
    ++found_count;
  }

  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(map_depthwise_conv_to_conv_pass,
              paddle::framework::ir::MapDepthwiseConv2ConvPass);
REGISTER_PASS_CAPABILITY(map_depthwise_conv_to_conv_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("depthwise_conv2d", 1)
            .LE("conv2d", 1));
