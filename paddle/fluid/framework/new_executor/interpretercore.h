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
#include "paddle/fluid/framework/new_executor/event_manager.h"
#include "paddle/fluid/framework/new_executor/interpretercore_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/new_executor/stream_analyzer.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {
using AtomicVectorSizeT = std::vector<std::unique_ptr<std::atomic<size_t>>>;

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place, const BlockDesc& block,
                  VariableScope* global_scope);

  ~InterpreterCore();

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors);

  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names);

  interpreter::CostInfo DryRun(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors);

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog);

 private:
  void Convert(std::vector<paddle::framework::OpFuncNode>* op_func_nodes);

  void BuildAndCacheInstructionCtx(Instruction* instr_node);

  void BuildInplace();

  bool BuildInplaceCheckVarIsOnlyInput(size_t var_index);

  void RunInstruction(const Instruction& instr_node);

  void ExecuteInstructionList(const std::vector<Instruction>& vec_instr);

  void Prepare(const std::vector<std::string>& feed_names,
               const std::vector<framework::LoDTensor>& feed_tensors,
               bool prepare_feed);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void RecordStreamForGC(const Instruction& instr);
#endif

  void CheckGC(const Instruction& instr);

  void RunInstructionAsync(size_t instr_id);
  void RunNextInstructions(const Instruction& instr_id,
                           std::queue<size_t>* reserved_next_ops);

  void BuildSkipShareLoDInfo();

  void BuildOperatorDependences();

  void SetFeedVarsInplaceSkip(const std::vector<std::string>& feed_names);

  void ClearLoDTensorArrayInLocalScope();

  bool is_build_;

  const platform::Place& place_;
  const BlockDesc& block_;  // not owned
  // NOTE(zhiqiu): when add fetch ops in GetInterpreterCore, we will
  // copy a new program and block, the copy_program_ here is used to
  // hold the program, otherwise block_ maybe not valid after the
  // new program is deleted.
  std::shared_ptr<ProgramDesc> copy_program_{nullptr};

  VariableScope* global_scope_;  // not owned

  std::vector<Instruction> vec_instruction_;  // deconstruct before OpFuncNode

  std::vector<size_t> dependecy_count_;
  std::atomic<size_t> unfinished_op_numer_{0};
  std::vector<std::vector<size_t>> input_var2op_info_;

  StreamAnalyzer stream_analyzer_;
  EventsWaiter main_thread_blocker_;
  std::unique_ptr<interpreter::AsyncWorkQueue> async_work_queue_;
  details::ExceptionHolder exception_holder_;
  std::shared_ptr<EventsWaiter::EventNotifier> exception_notifier_{nullptr};
  std::shared_ptr<EventsWaiter::EventNotifier> completion_notifier_{nullptr};

  std::unique_ptr<InterpreterCoreGarbageCollector> gc_;
  std::vector<paddle::platform::DeviceEvent> gc_event_;
  bool create_local_scope_{true};
  Scope* local_scope_{nullptr};  // not owned
};
}  // namespace framework
}  // namespace paddle
