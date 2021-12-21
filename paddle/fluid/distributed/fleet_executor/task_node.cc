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
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {
namespace {
using OperatorBase = TaskNode::OperatorBase;
}

TaskNode::TaskNode(paddle::framework::ProgramDesc* program, int64_t rank,
                   int64_t max_run_times, int64_t max_slot_nums)
    : program_(program),
      rank_(rank),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {
  // Should be serially invoked, not thread-safe
  // NOTE: when instantiate TaskNode with program, won't init task node
  // immediately, since the provided program may be updated later (with
  // high probability) by adding_feed_fetch_ops or by RuntimeGraph.
  // So, delay the init part to the Init() function.
  static int64_t task_node_cnt = 0;
  task_id_ = task_node_cnt++;
}

void TaskNode::SetProgram(paddle::framework::ProgramDesc* program) {
  program_ = program;
}

void TaskNode::Init() {
  if (ops_.empty()) {
    // Q (for fleet executor dev): should we need another reset funct?
    VLOG(3) << "Task node will be inited by calling Init().";
    for (const auto& op_desc : program_->Block(0).AllOps()) {
      ops_vec_.emplace_back(framework::OpRegistry::CreateOp(*op_desc));
    }
    for (const auto& op : ops_vec_) {
      ops_.emplace_back(op.get());
    }
  }
}

TaskNode::TaskNode(int32_t role,
                   const std::vector<framework::OpDesc*>& op_descs,
                   int64_t rank, int64_t task_id, int64_t max_run_times,
                   int64_t max_slot_nums)
    : role_(role),
      rank_(rank),
      task_id_(task_id),
      max_run_times_(max_run_times),
      max_slot_nums_(max_slot_nums) {
  if (op_descs.empty()) {
    return;
  }
  VLOG(3) << "Task node will be inited by providing list of ops.";
  for (const auto& desc : op_descs) {
    ops_vec_.emplace_back(framework::OpRegistry::CreateOp(*desc));
  }
  for (const auto& op : ops_vec_) {
    ops_.emplace_back(op.get());
  }
}

TaskNode::TaskNode(int32_t role,
                   const std::vector<framework::OperatorBase*>& ops,
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

bool TaskNode::AddUpstreamTask(int64_t task_id, int64_t buff_size) {
  const auto& ret = upstream_.emplace(task_id, buff_size);
  return ret.second;
}

bool TaskNode::AddDownstreamTask(int64_t task_id, int64_t buff_size) {
  const auto& ret = downstream_.emplace(task_id, buff_size);
  return ret.second;
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

void TaskNode::SetRunPerSteps(int64_t value) {
  PADDLE_ENFORCE_GE(value, 1,
                    platform::errors::InvalidArgument(
                        "run_per_steps must >= 1, but received %ld", value));
  run_per_steps_ = value;
}

void TaskNode::SetRunAtOffset(int64_t value) {
  PADDLE_ENFORCE_GE(value, 0,
                    platform::errors::InvalidArgument(
                        "run_at_offset must >= 0, but received %ld", value));
  run_at_offset_ = value;
}

void TaskNode::SetReplyUpPerSteps(int64_t value) {
  PADDLE_ENFORCE_GE(
      value, 1, platform::errors::InvalidArgument(
                    "reply_up_per_steps must >= 1, but received %ld", value));
  reply_up_per_steps_ = value;
}

void TaskNode::SetSendDownPerSteps(int64_t value) {
  PADDLE_ENFORCE_GE(
      value, 1, platform::errors::InvalidArgument(
                    "send_down_per_steps must >= 1, but received %ld", value));
  send_down_per_steps_ = value;
}

}  // namespace distributed
}  // namespace paddle
