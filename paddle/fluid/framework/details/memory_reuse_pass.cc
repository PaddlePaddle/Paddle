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

#include "paddle/fluid/framework/details/memory_optimize_pass.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>
#include "glog/logging.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> MemoryOptimizePass::ApplyImpl(std::unique_ptr<ir::Graph> graph) const {
  auto& node_map = Get<ReusedNodePairMap>(kGlobalReusedNodePairMap);
  auto update_graph_from_reuse_map = [&](Node* op, const ReusedNodePairMap& node_map) {
    auto* desc = op->Op();
  };
}


}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(memory_optimize_pass,
              paddle::framework::details::MemoryOptimizePass)
.RequirePassAttr(paddle::framework::details::kGlobalUnlivedNodePool);
