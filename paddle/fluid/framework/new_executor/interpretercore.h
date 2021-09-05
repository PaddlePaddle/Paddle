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

#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/new_executor/workqueue.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {

class InterpreterCore {
 public:
  using GarbageQueue = std::deque<std::shared_ptr<memory::Allocation>>;
  InterpreterCore(const platform::Place& place, const ProgramDesc& main_prog,
                  VariableScope* global_scope,
                  const std::vector<std::string>& feed_names,
                  const std::vector<std::string>& fetch_names);

  paddle::framework::FetchList Run(
      const std::vector<framework::Tensor>& feed_tensors);

  const CostInfo& DryRun(const std::vector<framework::Tensor>& feed_tensors);

  static void BuildOpFuncList(const platform::Place& place,
                              const framework::ProgramDesc& pdesc,
                              std::vector<OperatorBase*>* op_list,
                              std::vector<OpFuncNode>* vec_func_list,
                              VariableScope* var_scope);

 private:
  void Convert();

  void BuildInstructionCtx(Instruction* instr_node,
                           const VariableScope& var_scope,
                           const platform::Place& place);

  void RunInstruction(const Instruction& instr_node);

  void ExecuteInstructionList(const std::vector<Instruction>& vec_instr,
                              const VariableScope& var_scope,
                              const platform::Place& place,
                              bool is_dry_run = false);

  std::vector<size_t> MergeVector(const std::vector<size_t>& first,
                                  const std::vector<size_t>& second);

  void BuildVariableScope(const framework::ProgramDesc& pdesc,
                          VariableScope* var_scope);

  void Prepare(const std::vector<framework::Tensor>& feed_tensors);

  void CheckGC(size_t instr_id, const std::vector<size_t>& gc_check_list,
               const VariableScope& var_scope, const platform::Place& place,
               std::vector<VariableMetaInfo>& working_var_ref);  // NOLINT

  platform::DeviceContext* ParseDeviceContextForInstruction(
      const OpFuncNode& op_func_node, const OperatorBase& op_base);

  void RecordEventInstruction(const Instruction& instruction,
                              const OpFuncNode& op_func_node);

  void WaitOrSync(const std::vector<EventInter>& events,
                  const platform::DeviceContext* dev_ctx);

  void StreamWaitEventOrSync(const Instruction& instruction);

  const platform::Place& place_;
  ProgramDesc main_program_;
  VariableScope* global_scope_;
  platform::DeviceContextPool d2h_ctx_pool_;
  platform::DeviceContextPool h2d_ctx_pool_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  std::vector<paddle::framework::OpFuncNode> vec_func_list_;
  std::vector<paddle::framework::OperatorBase*> op_list_;

  std::vector<Instruction> vec_instruction_;
  InstructionInfo instruction_info_;
  std::vector<size_t> dependecy_count_;
  std::vector<VariableMetaInfo> ref_coun_info_;
  std::vector<std::vector<size_t>> input_var2op_info_;

  bool is_build_;

  std::vector<std::string> feed_names_;
  InterpreterProfiler profiler_;
  std::map<size_t, std::shared_ptr<platform::DeviceEvent>> var_id2event_;

  std::vector<paddle::platform::DeviceEvent> gc_event_;
  std::unique_ptr<GarbageQueue> garbages_;
  size_t max_memory_size_;
  size_t cur_memory_size_;
  std::unique_ptr<WorkQueue> gc_queue_;

  platform::DeviceContextPool fetch_context_pool_;
};
}  // namespace framework
}  // namespace paddle
