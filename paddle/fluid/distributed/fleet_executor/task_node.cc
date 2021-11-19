// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {
namespace {
using OperatorBase = TaskNode::OperatorBase;
}

TaskNode::TaskNode(int64_t role, const std::vector<OperatorBase*>& ops,
                   int64_t rank, int64_t task_id, int64_t max_run_times,
                   int64_t max_slot_nums)
    : ops_(ops),
      role_(role),
      rank_(rank),
      task_id_(task_id),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {}

TaskNode::TaskNode(int64_t role, int64_t rank, int64_t task_id,
                   int64_t max_run_times, int64_t max_slot_nums)
    : role_(role),
      rank_(rank),
      task_id_(task_id),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {}

std::unique_ptr<TaskNode> TaskNode::CreateEmptyTaskNode(int64_t role,
                                                        int64_t rank,
                                                        int64_t task_id,
                                                        int64_t max_run_times,
                                                        int64_t max_slot_nums) {
  return std::make_unique<TaskNode>(role, rank, task_id, max_run_times,
                                    max_slot_nums);
}

std::unique_ptr<TaskNode> TaskNode::CreateTaskNode(
    int64_t role, const std::vector<OperatorBase*>& ops, int64_t rank,
    int64_t task_id, int64_t max_run_times, int64_t max_slot_nums) {
  return std::make_unique<TaskNode>(role, ops, rank, task_id, max_run_times,
                                    max_slot_nums);
}

void TaskNode::AddUpstreamTask(int64_t task_id) { upstream_.insert(task_id); }

void TaskNode::AddDownstreamTask(int64_t task_id) {
  downstream_.insert(task_id);
}
}  // namespace distributed
}  // namespace paddle
