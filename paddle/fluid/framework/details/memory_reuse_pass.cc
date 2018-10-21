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

#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/details/memory_reuse_pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace details {

void MemoryReusePass::UpdateGraphFromReuseMap(
    const size_t& idx, const std::vector<ir::Node*>& graph_ops,
    ReusedNodePairMap* reused_node_map) const {
  // update op desc and op node from op idx to the end
  auto* start_op = graph_ops[idx];
  auto* var = reused_node_map->at(start_op).first;
  auto* cache_var = reused_node_map->at(start_op).second;
  for (size_t i = idx; i < graph_ops.size(); ++i) {
    auto* op = graph_ops[i];
    auto* op_desc = op->Op();
    // update desc
    op_desc->Rename(var->Name(), cache_var->Name());
    // update node
    std::replace(op->inputs.begin(), op->inputs.end(), var, cache_var);
    std::replace(op->outputs.begin(), op->outputs.end(), var, cache_var);
  }
  reused_node_map->erase(start_op);
}

std::unique_ptr<ir::Graph> MemoryReusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& reused_node_map = Get<ReusedNodePairMap>(kGlobalReusedNodePairMap);
  auto& graph_ops = Get<std::vector<ir::Node*>>(kGraphReusedOps);

  for (size_t i = 0; i < graph_ops.size(); ++i) {
    if (reused_node_map.count(graph_ops[i])) {
      UpdateGraphFromReuseMap(i, graph_ops, &reused_node_map);
    }
  }
  PADDLE_ENFORCE(reused_node_map.empty(), "Unmatched reused map and graph.");
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(memory_optimize_pass, paddle::framework::details::MemoryReusePass)
    .RequirePassAttr(paddle::framework::details::kGlobalReusedNodePairMap)
    .RequirePassAttr(paddle::framework::details::kGraphReusedOps);
