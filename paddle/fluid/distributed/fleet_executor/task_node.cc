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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {
namespace {
using OperatorBase = TaskNode::OperatorBase;
}

TaskNode::TaskNode(const framework::ProgramDesc& program, int64_t rank,
                   int64_t max_run_times, int64_t max_slot_nums)
    : program_(program),
      rank_(rank),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {
  // Should be serially invoked, not thread-safe
  static int64_t task_node_cnt = 0;
  task_id_ = task_node_cnt++;
  for (const auto& op_desc : program.Block(0).AllOps()) {
    ops_vec_.emplace_back(framework::OpRegistry::CreateOp(*op_desc));
  }
  for (const auto& op : ops_vec_) {
    ops_.emplace_back(op.get());
  }
}

TaskNode::TaskNode(int32_t role, const std::vector<OperatorBase*>& ops,
                   int64_t rank, int64_t task_id, int64_t max_run_times,
                   int64_t max_slot_nums)
    : ops_(ops),
      role_(role),
      rank_(rank),
      task_id_(task_id),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {}

TaskNode::TaskNode(int32_t role, int64_t rank, int64_t task_id,
                   int64_t max_run_times, int64_t max_slot_nums)
    : role_(role),
      rank_(rank),
      task_id_(task_id),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {}

std::unique_ptr<TaskNode> TaskNode::CreateEmptyTaskNode(int32_t role,
                                                        int64_t rank,
                                                        int64_t task_id,
                                                        int64_t max_run_times,
                                                        int64_t max_slot_nums) {
  return std::make_unique<TaskNode>(role, rank, task_id, max_run_times,
                                    max_slot_nums);
}

std::unique_ptr<TaskNode> TaskNode::CreateTaskNode(
    int32_t role, const std::vector<OperatorBase*>& ops, int64_t rank,
    int64_t task_id, int64_t max_run_times, int64_t max_slot_nums) {
  return std::make_unique<TaskNode>(role, ops, rank, task_id, max_run_times,
                                    max_slot_nums);
}

bool TaskNode::AddUpstreamTask(int64_t task_id) {
  const auto& ret = upstream_.insert(task_id);
  return *ret.first == task_id;
}

bool TaskNode::AddDownstreamTask(int64_t task_id) {
  const auto& ret = downstream_.insert(task_id);
  return *ret.first == task_id;
}

std::string TaskNode::DebugString() const {
  std::ostringstream os;
  os << "role: " << role_ << ", task_id: " << task_id_ << "\n";
  for (std::size_t i = 0; i < ops_.size(); ++i) {
    os << ops_[i]->Type() << " ";
  }
  os << "\n";
  return os.str();
}
}  // namespace distributed
}  // namespace paddle
