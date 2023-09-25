// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter_base_impl.h"
#include "paddle/pir/core/value.h"

namespace ir {
class Block;
}  // namespace ir

namespace paddle {
namespace framework {
class ValueExecutionInfo;
class NewIRInterpreter : public InterpreterBaseImpl {
  using ExecutionConfig = interpreter::ExecutionConfig;
  using InstructionSchedulingPriorityLess = std::function<bool(size_t, size_t)>;
  using SchedulingQueue =
      std::priority_queue<size_t,
                          std::vector<size_t>,
                          InstructionSchedulingPriorityLess>;

 public:
  NewIRInterpreter(const platform::Place& place,
                   const std::vector<std::string>& fetch_var_names,
                   const ::pir::Block* ir_block,
                   Scope* scope,
                   const ExecutionConfig& execution_config = ExecutionConfig());

  NewIRInterpreter(const platform::Place& place,
                   const std::vector<std::string>& fetch_var_names,
                   const ::pir::Block* ir_block,
                   Scope* scope,
                   std::shared_ptr<ValueExecutionInfo> value_exe_info,
                   const ExecutionConfig& execution_config = ExecutionConfig());

  ~NewIRInterpreter();

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors) override;

  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names,
                                   bool need_fetch = true) override;

  void ShareWorkQueueFrom(InterpreterBaseImpl* src) override;

  void ShareBuildResultsFrom(const InterpreterBaseImpl& src) override;

  std::shared_ptr<std::vector<size_t>> GetDependencyCount() const override;

  bool IsSharedResultsBuild() const override;

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog) override;

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) override;

  const std::set<std::string>& JitInputVars() const override;

  void SetJitInputVars(const std::set<std::string>& jit_input_vars) override;

  const VariableScope* GetVariableScope() const override;

  void reset_scope(Scope* new_scope) override;

  const Scope* local_scope() const override;

  const platform::Place& GetPlace() const override { return place_; }

  void SetOutputHooks(const std::vector<HookFunc>& hookfuncs) override {
    hookfuncs_ = hookfuncs;
  }

  std::string GetNameByValue(::pir::Value value) const;

 private:
  // build graph
  void UpdateSyncOpNum();
  void UpdateNcclOpNum();
  void AnalyseExecuteOrderForTrace(
      std::map<size_t, std::set<size_t>> op_downstream_map,
      InstructionSchedulingPriorityLess compare);
  void ConstructEventForJitInput();
  void CalculateLastLiveOps();

  // gc
  void ClearLoDTensorArrayInLocalScope();

  // cuda graph
  void CheckCUDAGraphBeforeRun(const std::vector<std::string>& feed_names);
  void PrepareForCUDAGraphCapture();

  void Build(
      const std::vector<std::string>& feed_names,
      std::vector<paddle::framework::OpFuncNode>* op_func_nodes) override;

  bool IsStaticBuild() const override { return static_build_; }

  // workqueue
  std::shared_ptr<interpreter::AsyncWorkQueue> GetWorkQueue();

  // scope
  bool HasLocalScope() const;

  Scope* InnerScope();

  // For log and debug
  std::string GetDepsString() const;

  bool is_build_{false};
  bool static_build_{false};

  // Note(sonder): share the op dependency and event analysis procedure.
  bool is_shared_results_build_{false};

  const platform::Place place_;

  // from variable scope

  std::atomic<size_t> unfinished_op_number_{0};

  ExecutionConfig execution_config_;

  VariableScope var_scope_;
  Scope* scope_{nullptr};
  Scope* local_scope_{nullptr};  // not owned

  EventsWaiter main_thread_blocker_;
  std::shared_ptr<interpreter::AsyncWorkQueue> async_work_queue_;

  details::ExceptionHolder exception_holder_;
  std::shared_ptr<EventsWaiter::EventNotifier> exception_notifier_{nullptr};
  std::shared_ptr<EventsWaiter::EventNotifier> completion_notifier_{nullptr};

  std::unique_ptr<InterpreterCoreGarbageCollector> gc_;

  // last_live_ops_[i] contains the id of operators that last access the i-th
  // var
  std::map<size_t, std::set<size_t>> last_live_ops_;

  // (*dependecy_count_)[i] contains the number of dependencies that the i-th op
  // need to wait
  std::shared_ptr<std::vector<size_t>> dependecy_count_;

  std::vector<std::shared_ptr<interpreter::OpDepInfo>> deps_;
  std::vector<std::shared_ptr<interpreter::VarRefInfo>> refs_;

  // used for Trace
  int64_t sync_op_num_{-1};
  int64_t nccl_op_num_{-1};
  std::vector<size_t> trace_execute_order_;

  std::vector<HookFunc> hookfuncs_;

  /// ======================== ///
  ///        For new ir        ///
  /// ======================== ///
  std::string DebugValueInfo();

  void PreAnalysis();

  void BuildInstruction();

  void BuildInstructionDependences();

  void TraceRunImpl();

  void TraceRunInstructionList(
      const std::vector<std::unique_ptr<InstructionBase>>& vec_instr);

  void MultiThreadRunImpl();

  void MultiThreadRunInstructionList(
      const std::vector<std::unique_ptr<InstructionBase>>& vec_instr);

  void RunInstructionBaseAsync(size_t instr_id);

  void RunNextInstructions(InstructionBase* instr,
                           SchedulingQueue* reserved_next_ops);

  void RunInstructionBase(InstructionBase* instr_node);

  void RecordMemcpyD2H(InstructionBase* instr_node);

  ::pir::Value GetValueByName(const std::string& var_name);

  void CheckGC(InstructionBase* instr);

  void RecordStreamForGC(InstructionBase* instr);

  void SolvePersisableVarNames();

  const interpreter::NewIrDependencyBuilder& GetNewIrDependencyBuilder() const;

  const interpreter::NewIrStreamAnalyzer& GetNewIrStreamAnalyzer() const;

  InstructionSchedulingPriorityLess ir_instruction_scheduling_priority_less;

  const ::pir::Block* ir_block_{nullptr};

  std::vector<std::unique_ptr<InstructionBase>> vec_instruction_base_;

  // value execution info
  std::shared_ptr<ValueExecutionInfo> value_exe_info_;

  std::map<pir::Block*, paddle::framework::Scope*> sub_blocks_;

  std::vector<int> var_ref_count_;

  interpreter::NewIrDependencyBuilder ir_dependency_builder_;

  interpreter::NewIrStreamAnalyzer ir_stream_analyzer_;

  std::vector<std::string> fetch_var_names_;

  // Note(zhangbo): set_parameter_op's input and get_parameter_op's output
  // belongs to a parameter and cannot GC.
  std::unordered_set<std::string> parameter_var_names_;
};

}  // namespace framework
}  // namespace paddle
