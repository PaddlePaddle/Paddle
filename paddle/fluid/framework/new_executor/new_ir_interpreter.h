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
#include "paddle/ir/core/value.h"

namespace ir {
class Program;
}  // namespace ir

namespace paddle {
namespace framework {

class NewIRInterpreter : public InterpreterBaseImpl {
  using ExecutionConfig = interpreter::ExecutionConfig;
  using InstructionSchedulingPriorityLess = std::function<bool(size_t, size_t)>;
  using SchedulingQueue =
      std::priority_queue<size_t,
                          std::vector<size_t>,
                          InstructionSchedulingPriorityLess>;

 public:
  NewIRInterpreter(const platform::Place& place,
                   std::unique_ptr<::ir::Program> ir_prog,
                   Scope* scope,
                   const ExecutionConfig& execution_config = ExecutionConfig());

  ~NewIRInterpreter();

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors) override;

  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names,
                                   bool need_fetch = true) override;

  paddle::framework::FetchList BetaRun(
      const std::vector<std::string>& feed_names,
      bool need_fetch = true) override;

  void ShareWorkQueueFrom(InterpreterBaseImpl* src) override;

  void ShareBuildResultsFrom(const InterpreterBaseImpl& src) override;

  // op dependences
  const interpreter::DependencyBuilder& GetDependencyBuilder() const override;

  std::shared_ptr<std::vector<size_t>> GetDependencyCount() const override;

  const interpreter::StreamAnalyzer& GetStreamAnalyzer() const override;

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

  std::string GetNameById(int id) const;

  int GetIdByName(const std::string& name) const;

 private:
  // build graph
  void Convert(std::vector<paddle::framework::OpFuncNode>* op_func_nodes);
  void BuildOperatorDependences();
  void BuildAndCacheInstructionCtx(Instruction* instr_node);
  void BuildSkipShareLoDInfo();
  void UpdateSyncOpNum();
  void AnalyseExecuteOrderForTrace(
      std::map<size_t, std::set<size_t>> op_downstream_map,
      InstructionSchedulingPriorityLess compare);
  void ConstructEventForJitInput();
  void CalculateLastLiveOps();

  // inplace
  void BuildInplace();
  bool BuildInplaceCheckVarIsOnlyInput(
      const std::vector<std::vector<size_t>>& input_var2op, size_t var_index);
  void SetFeedVarsInplaceSkip(const std::vector<std::string>& feed_names);

  // cuda graph
  void CheckCUDAGraphBeforeRun(const std::vector<std::string>& feed_names);
  void PrepareForCUDAGraphCapture();

  // execution
  void RunImpl();
  void ExecuteInstructionList(const std::vector<Instruction>& vec_instr);
  void RunInstructionAsync(size_t instr_id);
  void RunInstruction(const Instruction& instr_node);
  void RunNextInstructions(const Instruction& instr_id,
                           SchedulingQueue* reserved_next_ops);
  void RunOperator(const Instruction& instr_node);
  // Trace
  void TraceInstructionList(const std::vector<Instruction>& vec_instr);

  // only used when program contains no feed op
  void Prepare(const std::vector<std::string>& feed_names,
               const std::vector<phi::DenseTensor>& feed_tensors,
               bool prepare_feed);

  void RecordMemcpyD2H(const Instruction& instr_node);

  // gc
  void RecordStreamForGC(const Instruction& instr);
  void CheckGC(const Instruction& instr);
  void ClearLoDTensorArrayInLocalScope();

  // workqueue
  std::shared_ptr<interpreter::AsyncWorkQueue> GetWorkQueue();

  // scope
  bool HasLocalScope() const;

  Scope* InnerScope();

  // For log and debug
  std::string GetDepsString() const;

  bool is_build_{false};
  bool static_build_{false};

  const platform::Place place_;

  interpreter::DependencyBuilder dependency_builder_;
  interpreter::StreamAnalyzer stream_analyzer_;

  // NOTE(zhiqiu): when add fetch ops in GetInterpreterCore, we will
  // copy a new program and block, the copy_program_ here is used to
  // hold the program, otherwise block_ maybe not valid after the
  // new program is deleted.
  std::shared_ptr<ProgramDesc> copy_program_{nullptr};

  // from variable scope
  std::vector<Variable*> var_list_;
  std::map<std::string, int> name2id_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  std::vector<Instruction> vec_instruction_;  // deconstruct before OpFuncNode

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

  // dependecy_count_[i] contains the number of dependencies that the i-th op
  // need to wait
  std::vector<size_t> dependecy_count_;

  std::vector<std::shared_ptr<interpreter::OpDepInfo>> deps_;
  std::vector<std::shared_ptr<interpreter::VarRefInfo>> refs_;

  // used for Trace
  int64_t sync_op_num_{-1};
  std::vector<size_t> trace_execute_order_;

  InstructionSchedulingPriorityLess instruction_scheduling_priority_less;

  std::vector<HookFunc> hookfuncs_;

  /// ======================== ///
  ///        For new ir        ///
  /// ======================== ///
  std::string DebugValueInfo();

  void PreAnalysis();

  void BuildInstruction();

  void BuildInstructionDependences();

  void NewIrLoopRunImpl();

  void BetaRunImpl();

  void TraceInstructionList(
      const std::vector<std::unique_ptr<InstructionBase>>& vec_instr);

  void RunInstructionBase(InstructionBase* instr_node);

  void RecordMemcpyD2H(InstructionBase* instr_node);

  ::ir::Value GetValueByName(const std::string& var_name);

  void CheckGC(InstructionBase* instr);

  void RecordStreamForGC(InstructionBase* instr);

  InstructionSchedulingPriorityLess ir_instruction_scheduling_priority_less;

  std::unique_ptr<::ir::Program> ir_program_{nullptr};

  std::vector<std::unique_ptr<InstructionBase>> vec_instruction_base_;

  std::unordered_map<::ir::Value, std::string> value_2_var_name_;

  std::unordered_map<const paddle::framework::Variable*, std::string>
      variable_2_var_name_;

  std::map<std::string, int> var_name_2_id_;

  std::vector<Variable*> variable_list_;

  std::vector<int> var_ref_count_;

  interpreter::NewIrDependencyBuilder ir_dependency_builder_;

  interpreter::NewIrStreamAnalyzer ir_stream_analyzer_;
};

}  // namespace framework
}  // namespace paddle
