// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/poly/naive_scheduler.h"
#include <vector>
#include "paddle/common/enforce.h"

namespace cinn {
namespace poly {

std::unique_ptr<Schedule> NaiveScheduler::BuildSchedule() {
  PartitionGroups();
  PADDLE_ENFORCE_EQ(!groups_.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The groups are empty. Please provide valid groups."));

  for (auto &group : groups_) {
    std::vector<Stage *> status;
    PADDLE_ENFORCE_EQ(
        group.nodes.size(),
        1UL,
        ::common::errors::InvalidArgument(
            "group.nodes.size() should be 1, but got %d", group.nodes.size()));
    NaiveGroupScheduler scheduler(
        const_cast<Stage *>(group.nodes.front()->stage));
    scheduler.Build();
  }

  std::unique_ptr<Schedule> res(new Schedule);
  res->groups = groups_;

  return res;
}

void NaiveScheduler::PartitionGroups() {
  // treat each node as a unique group, collect the groups in topological order.
  auto topo_order = schedule_graph_.topological_order();  // NOLINT
  auto &nodes_in_order = std::get<0>(topo_order);
  auto &edges_in_order = std::get<1>(topo_order);

  for (auto *node : nodes_in_order) {
    ScheduleGroup group;
    group.nodes.push_back(node->safe_as<ScheduleGraphNode>());
    groups_.emplace_back(std::move(group));
  }
}

}  // namespace poly
}  // namespace cinn
