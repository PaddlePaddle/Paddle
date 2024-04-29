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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class OperatorBase;
class OpDesc;
}  // namespace framework
namespace distributed {

enum class DependType { NORMAL, LOOP, STOP_LOOP };

class TaskNode final {
 public:
  using OperatorBase = paddle::framework::OperatorBase;
  TaskNode(int64_t rank, int64_t task_id, int64_t max_run_times);
  TaskNode(int32_t role, int64_t rank, int64_t task_id, int64_t max_run_times);
  TaskNode(int32_t role,
           const std::vector<framework::OpDesc*>& op_descs,
           int64_t rank,
           int64_t task_id,
           int64_t max_run_times);
  TaskNode(int32_t role,
           const std::vector<framework::OperatorBase*>& ops,
           int64_t rank,
           int64_t task_id,
           int64_t max_run_times);
  TaskNode(paddle::framework::ProgramDesc* program, int64_t rank);
  // TODO(liyurui): This will be the only constructor for task node
  TaskNode(paddle::framework::ProgramDesc* program,
           int64_t task_id,
           int64_t rank,
           int64_t max_run_times);

  ~TaskNode() = default;

  void SetProgram(paddle::framework::ProgramDesc* program);
  void Init(bool use_feed_fetch_ops = true);
  int64_t rank() const { return rank_; }
  int64_t task_id() const { return task_id_; }
  int32_t role() const { return role_; }
  int64_t max_run_times() const { return max_run_times_; }
  int64_t run_per_steps() const { return run_per_steps_; }
  int64_t run_at_offset() const { return run_at_offset_; }
  int64_t reply_up_per_steps() const { return reply_up_per_steps_; }
  int64_t send_down_per_steps() const { return send_down_per_steps_; }
  const std::string& cond_var() const { return cond_var_; }
  const std::unordered_map<int64_t, int64_t>& upstream() const {
    return upstream_;
  }
  const std::unordered_map<int64_t, int64_t>& downstream() const {
    return downstream_;
  }
  const std::string& type() const { return type_; }
  const paddle::framework::ProgramDesc* program() const { return program_; }
  const std::vector<OperatorBase*>& ops() const { return ops_; }
  const std::vector<std::unique_ptr<OperatorBase>>& unique_ops() const {
    return ops_vec_;
  }
  const std::unordered_map<int64_t, DependType> id_to_dep_type() const {
    return id_to_dep_type_;
  }
  const std::unordered_map<const OperatorBase*, std::vector<std::string>>&
  unused_vars() const {
    return unused_vars_;
  }
  const std::vector<std::string> while_block_vars() const {
    return while_block_vars_;
  }

  void SetCondVarName(const std::string& cond_var_name) {
    cond_var_ = cond_var_name;
  }
  void SetRunPerSteps(int64_t value);
  void SetRunAtOffset(int64_t value);
  void SetReplyUpPerSteps(int64_t value);
  void SetSendDownPerSteps(int64_t value);
  void SetType(const std::string& type) { type_ = type; }
  void SetUnusedVars(
      const std::unordered_map<const OperatorBase*, std::vector<std::string>>&
          unused_vars) {
    unused_vars_ = unused_vars;
  }
  void SetWhileBlockVars(const std::vector<std::string>& vars) {
    while_block_vars_ = vars;
  }

  // upstream need buffs?
  bool AddUpstreamTask(int64_t task_id,
                       int64_t buff_size = 1,
                       DependType type = DependType::NORMAL);
  bool AddDownstreamTask(int64_t task_id,
                         int64_t buff_size = 1,
                         DependType type = DependType::NORMAL);
  std::string DebugString() const;
  void SetVarsToDtype(const std::map<std::string, std::string>& vars_to_dtype);
  const std::map<std::string, std::vector<int64_t>>& vars_to_shape() const {
    return vars_to_shape_;
  }
  const std::map<std::string, std::string>& vars_to_dtype() const {
    return vars_to_dtype_;
  }
  void SetVarsToShape(
      const std::map<std::string, std::vector<int64_t>>& vars_to_shape);

 private:
  DISABLE_COPY_AND_ASSIGN(TaskNode);
  TaskNode() = default;
  // ops_ will be removed in the future
  std::vector<OperatorBase*> ops_;
  // task_id-->buff_size
  std::unordered_map<int64_t, int64_t> upstream_;
  std::unordered_map<int64_t, int64_t> downstream_;
  // task_id-->type
  std::unordered_map<int64_t, DependType> id_to_dep_type_;

  framework::ProgramDesc* program_{nullptr};
  std::string cond_var_;
  std::vector<std::unique_ptr<OperatorBase>> ops_vec_;
  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
  std::vector<std::string> while_block_vars_;

  int32_t role_;
  int64_t rank_;
  int64_t task_id_;
  int64_t max_run_times_;

  int64_t run_per_steps_{1};
  int64_t run_at_offset_{0};
  // one input produces multi times output
  int64_t reply_up_per_steps_{1};
  // one output need multi times input
  int64_t send_down_per_steps_{1};

  std::string type_;
  std::map<std::string, std::string> vars_to_dtype_;
  std::map<std::string, std::vector<int64_t>> vars_to_shape_;
};

}  // namespace distributed
}  // namespace paddle
