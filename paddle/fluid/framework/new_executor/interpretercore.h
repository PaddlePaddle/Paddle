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

#include "paddle/fluid/framework/new_executor/event_manager.h"
#include "paddle/fluid/framework/new_executor/interpretercore_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/new_executor/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/workqueue.h"
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
  InterpreterCore(const platform::Place& place, const ProgramDesc& main_prog,
                  VariableScope* global_scope,
                  const std::vector<std::string>& feed_names,
                  const std::vector<std::string>& fetch_names);

  paddle::framework::FetchList Run(
      const std::vector<framework::Tensor>& feed_tensors);

  const CostInfo& DryRun(const std::vector<framework::Tensor>& feed_tensors);

 private:
  void Convert();

  void BuildAndCacheInstructionCtx(Instruction* instr_node,
                                   const VariableScope& var_scope,
                                   const platform::Place& place);

  void BuildInplace();

  bool BuildInplaceCheckVarIsOnlyInput(size_t var_index);

  void RunInstruction(const Instruction& instr_node);

  void ExecuteInstructionList(const std::vector<Instruction>& vec_instr);

  void DryRunPrepare(const std::vector<framework::Tensor>& feed_tensors);

  void CheckGC(size_t instr_id, const std::vector<size_t>& gc_check_list);

  void RunInstructionAsync(size_t instr_id);
  void RunNextInstruction(const Instruction& instr_id);
  void AddFetch(const std::vector<std::string>& fetch_names);

  void BuildSkipShareLoDInfo();

  bool is_build_;

  const platform::Place& place_;
  ProgramDesc main_program_;
  VariableScope* global_scope_;

  std::vector<Instruction> vec_instruction_;
  InstructionInfo instruction_info_;
  std::vector<size_t> dependecy_count_;
  std::vector<std::vector<size_t>> input_var2op_info_;
  std::vector<VariableMetaInfo> ref_coun_info_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  std::vector<paddle::framework::OpFuncNode> vec_func_list_;
  std::vector<paddle::framework::OperatorBase*> op_list_;

  std::vector<std::string> feed_names_;

  InterpreterProfiler dry_run_profiler_;
  StreamAnalyzer stream_analyzer_;
  EventManager event_manager_;
  interpretercore::AsyncWorkQueue async_work_queue_;

  InterpreterCoreGarbageCollector gc_;
  std::vector<paddle::platform::DeviceEvent> gc_event_;
  std::atomic<size_t> op_run_number_{0};
};
}  // namespace framework
}  // namespace paddle
