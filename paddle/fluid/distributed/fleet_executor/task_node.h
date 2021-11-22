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

#pragma once
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class OperatorBase;
}
namespace distributed {

class TaskNode final {
 public:
  using OperatorBase = paddle::framework::OperatorBase;
  TaskNode(int64_t role, int64_t rank, int64_t task_id);
  TaskNode(int64_t role, const std::vector<OperatorBase*>& ops, int64_t rank,
           int64_t task_id);
  ~TaskNode() = default;
  int64_t rank() const { return rank_; }
  int64_t task_id() const { return task_id_; }
  int64_t role() const { return role_; }
  const std::unordered_set<int64_t>& upstream() const { return upstream_; }
  const std::unordered_set<int64_t>& downstream() const { return downstream_; }
  void AddUpstreamTask(int64_t task_id);
  void AddDownstreamTask(int64_t task_id);
  static std::unique_ptr<TaskNode> CreateEmptyTaskNode(int64_t role,
                                                       int64_t rank,
                                                       int64_t task_id);
  static std::unique_ptr<TaskNode> CreateTaskNode(
      int64_t role, const std::vector<OperatorBase*>& ops, int64_t rank,
      int64_t task_id);

 private:
  DISABLE_COPY_AND_ASSIGN(TaskNode);
  TaskNode() = default;
  std::vector<OperatorBase*> ops_;
  std::unordered_set<int64_t> upstream_;
  std::unordered_set<int64_t> downstream_;
  int64_t role_;
  int64_t rank_;
  int64_t task_id_;
};

}  // namespace distributed
}  // namespace paddle
