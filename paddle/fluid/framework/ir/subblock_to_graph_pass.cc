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

#include "paddle/fluid/framework/ir/subblock_to_graph_pass.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> SubblockToGraphPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // Filter out the nodes that has sub-block
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->HasAttr(kSubblockAttrKeyStr)) {
      auto* sub_block_desc = boost::get<framework::BlockDesc*>(
          node->Op()->GetAttr(kSubblockAttrKeyStr));
      // sub-block to programdesc
      for (auto op : sub_block_desc->Proto()->ops()) {
        LOG(INFO) << op.type();
      }

      framework::ProgramDesc fake_program_desc;
    }
  }
  // Build a graph for the sub-block
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
