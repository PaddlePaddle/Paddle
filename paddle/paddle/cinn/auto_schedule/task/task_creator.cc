// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/task/task_creator.h"

#include <glog/logging.h>

#include <memory>
#include <tuple>
#include <vector>

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/pass.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::common::GraphEdge;
using ::cinn::common::GraphNode;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::Node;
using ::cinn::hlir::framework::NodeData;

std::vector<TuneTask> TaskCreator::CreateTuneTaskOpLevel(Graph* graph) {
  std::vector<TuneTask> ret_tasks;

  const std::vector<std::shared_ptr<Graph::Group>>* groups =
      &graph->fusion_groups;
  std::vector<std::shared_ptr<Graph::Group>> non_fused_groups;
  // The input graph doesn't run Op Fusion
  if (graph->fusion_groups.empty()) {
    hlir::framework::ApplyPasses(graph, {"BuildNonFusedGroupsPass"});
    groups = &graph->fusion_groups;
  }
  VLOG(3) << "Graph groups size:" << groups->size();

  for (const auto& sub_graph : *groups) {
    ret_tasks.emplace_back(TuneTask());
    ret_tasks.back().subgraph = sub_graph;
    ret_tasks.back().target = graph->target_;
  }
  return ret_tasks;
}

}  // namespace auto_schedule
}  // namespace cinn
