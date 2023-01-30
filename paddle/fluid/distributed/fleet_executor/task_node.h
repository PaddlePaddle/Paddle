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
<<<<<<< HEAD
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
=======
#include <memory>
#include <string>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
enum class DependType { NORMAL, LOOP, STOP_LOOP };

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
class TaskNode final {
 public:
  using OperatorBase = paddle::framework::OperatorBase;
  TaskNode(int64_t rank, int64_t task_id, int64_t max_run_times);
<<<<<<< HEAD
  TaskNode(int32_t role, int64_t rank, int64_t task_id, int64_t max_run_times);
=======
  TaskNode(int32_t role,
           int64_t rank,
           int64_t task_id,
           int64_t max_run_times,
           int64_t max_slot_nums);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  TaskNode(int32_t role,
           const std::vector<framework::OpDesc*>& op_descs,
           int64_t rank,
           int64_t task_id,
<<<<<<< HEAD
           int64_t max_run_times);
=======
           int64_t max_run_times,
           int64_t max_slot_nums);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  TaskNode(int32_t role,
           const std::vector<framework::OperatorBase*>& ops,
           int64_t rank,
           int64_t task_id,
<<<<<<< HEAD
           int64_t max_run_times);
=======
           int64_t max_run_times,
           int64_t max_slot_nums);
  TaskNode(paddle::framework::ProgramDesc* program,
           int64_t rank,
           int64_t max_run_times,
           int64_t max_slot_nums);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  TaskNode(paddle::framework::ProgramDesc* program, int64_t rank);
  // TODO(liyurui): This will be the only constructor for task node
  TaskNode(paddle::framework::ProgramDesc* program,
           int64_t task_id,
           int64_t rank,
<<<<<<< HEAD
           int64_t max_run_times);

=======
           int64_t max_run_times,
           int64_t max_slot_nums);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  ~TaskNode() = default;

  void SetProgram(paddle::framework::ProgramDesc* program);
  void Init(bool use_feed_fetch_ops = true);
  int64_t rank() const { return rank_; }
  int64_t task_id() const { return task_id_; }
  int32_t role() const { return role_; }
  int64_t max_run_times() const { return max_run_times_; }
<<<<<<< HEAD
=======
  int64_t max_slot_nums() const { return max_slot_nums_; }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  int64_t run_per_steps() const { return run_per_steps_; }
  int64_t run_at_offset() const { return run_at_offset_; }
  int64_t reply_up_per_steps() const { return reply_up_per_steps_; }
  int64_t send_down_per_steps() const { return send_down_per_steps_; }
<<<<<<< HEAD
  const std::string& cond_var() const { return cond_var_; }
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
  const std::unordered_map<int64_t, DependType> id_to_dep_type() const {
    return id_to_dep_type_;
  }
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  const std::unordered_map<const OperatorBase*, std::vector<std::string>>&
  unused_vars() const {
    return unused_vars_;
  }
<<<<<<< HEAD
  const std::vector<std::string> while_block_vars() const {
    return while_block_vars_;
  }

  void SetCondVarName(const std::string& cond_var_name) {
    cond_var_ = cond_var_name;
  }
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======

  // upstream need buffs?
  bool AddUpstreamTask(int64_t task_id, int64_t buff_size = 1);
  bool AddDownstreamTask(int64_t task_id, int64_t buff_size = 1);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  std::string DebugString() const;

 private:
  DISABLE_COPY_AND_ASSIGN(TaskNode);
  TaskNode() = default;
  // ops_ will be removed in the future
  std::vector<OperatorBase*> ops_;
  // task_id-->buff_size
  std::unordered_map<int64_t, int64_t> upstream_;
  std::unordered_map<int64_t, int64_t> downstream_;
<<<<<<< HEAD
  // task_id-->type
  std::unordered_map<int64_t, DependType> id_to_dep_type_;

  framework::ProgramDesc* program_;
  std::string cond_var_;
  std::vector<std::unique_ptr<OperatorBase>> ops_vec_;
  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
  std::vector<std::string> while_block_vars_;
=======
  framework::ProgramDesc* program_;
  std::vector<std::unique_ptr<OperatorBase>> ops_vec_;
  std::unordered_map<const OperatorBase*, std::vector<std::string>>
      unused_vars_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  int32_t role_;
  int64_t rank_;
  int64_t task_id_;
  int64_t max_run_times_;
<<<<<<< HEAD
=======
  int64_t max_slot_nums_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  int64_t run_per_steps_{1};
  int64_t run_at_offset_{0};
  // one input produces multi times output
  int64_t reply_up_per_steps_{1};
  // one output need multi times input
  int64_t send_down_per_steps_{1};

  std::string type_;
};

}  // namespace distributed
}  // namespace paddle
