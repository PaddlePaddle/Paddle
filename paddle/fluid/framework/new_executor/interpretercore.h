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

#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/framework/new_executor/interpreter/event_manager.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/new_executor/stream_analyzer.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/device_event.h"

DECLARE_bool(new_executor_use_local_scope);
DECLARE_bool(control_flow_use_new_executor);

namespace paddle {
namespace framework {

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place,
                  const BlockDesc& block,
                  const std::set<std::string>& skip_gc_vars,
                  Scope* scope,
                  bool used_for_jit = false,
                  bool used_for_control_flow_op = false);

  ~InterpreterCore();

  interpreter::CostInfo DryRun(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors);

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors);

  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names,
                                   bool need_fetch = true);

  void ShareWorkQueueFrom(std::shared_ptr<InterpreterCore> src);

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog);

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars);

  const VariableScope* GetVariableScope() const;

  void reset_scope(Scope* new_scope);

  const platform::Place& GetPlace() const { return place_; }

 private:
  // build graph
  void Convert(std::vector<paddle::framework::OpFuncNode>* op_func_nodes);
  void BuildOperatorDependences();
  void BuildAndCacheInstructionCtx(Instruction* instr_node);
  void BuildSkipShareLoDInfo();

  // inplace
  void BuildInplace();
  bool BuildInplaceCheckVarIsOnlyInput(
      const std::vector<std::vector<size_t>>& input_var2op, size_t var_index);
  void SetFeedVarsInplaceSkip(const std::vector<std::string>& feed_names);

  // execution
  void ExecuteInstructionList(const std::vector<Instruction>& vec_instr);
  void RunInstructionAsync(size_t instr_id);
  void RunInstruction(const Instruction& instr_node);
  void RunNextInstructions(const Instruction& instr_id,
                           std::deque<size_t>* reserved_next_ops);
  // only used when program contains no feed op
  void Prepare(const std::vector<std::string>& feed_names,
               const std::vector<phi::DenseTensor>& feed_tensors,
               bool prepare_feed);

  // gc
  void RecordStreamForGC(const Instruction& instr);
  void CheckGC(const Instruction& instr);
  void ClearLoDTensorArrayInLocalScope();

  // workqueue
  std::shared_ptr<interpreter::AsyncWorkQueue> GetWorkQueue();

  // scope
  bool HasLocalScope() const;

 private:
  bool is_build_{false};

  platform::Place place_;
  const BlockDesc& block_;  // not owned

  interpreter::DependencyBuilder dependency_builder_;
  interpreter::ExecutionConfig execution_config_;

  // NOTE(zhiqiu): when add fetch ops in GetInterpreterCore, we will
  // copy a new program and block, the copy_program_ here is used to
  //
  // hold the program, otherwise block_ maybe not valid after the
  // new program is deleted.
  std::shared_ptr<ProgramDesc> copy_program_{nullptr};

  // from variable scope
  std::vector<Variable*> var_list_;
  std::map<std::string, int> name2id_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  std::vector<Instruction> vec_instruction_;  // deconstruct before OpFuncNode

  std::atomic<size_t> unfinished_op_number_{0};
  VariableScope var_scope_;
  Scope* local_scope_{nullptr};  // not owned

  StreamAnalyzer stream_analyzer_;
  EventsWaiter main_thread_blocker_;
  std::shared_ptr<interpreter::AsyncWorkQueue> async_work_queue_;

  details::ExceptionHolder exception_holder_;
  std::shared_ptr<EventsWaiter::EventNotifier> exception_notifier_{nullptr};
  std::shared_ptr<EventsWaiter::EventNotifier> completion_notifier_{nullptr};

  std::unique_ptr<InterpreterCoreGarbageCollector> gc_;

  // last_live_ops_[i] contains the id of operators that last access the i-th
  // var
  std::map<size_t, std::set<size_t>> last_live_ops_;

  // dependecy_count_[i] contains the number of dependencies that the i-th op
  // need to wait
  std::vector<size_t> dependecy_count_;

  std::vector<std::shared_ptr<interpreter::OpDepInfo>> deps_;
  std::vector<std::shared_ptr<interpreter::VarRefInfo>> refs_;
};

std::shared_ptr<InterpreterCore> CreateInterpreterCore(
    const platform::Place& place,
    const ProgramDesc& prog,
    Scope* global_scope,
    const std::vector<std::string>& fetch_names = {},
    const std::set<std::string>& skip_gc_vars = {});

}  // namespace framework
}  // namespace paddle
