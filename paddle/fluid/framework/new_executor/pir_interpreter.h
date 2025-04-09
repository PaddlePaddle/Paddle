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
#include "paddle/pir/include/core/value.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#endif

namespace ir {
class Block;
}  // namespace ir

namespace paddle {
namespace framework {
class ValueExecutionInfo;
class PirInterpreter : public InterpreterBaseImpl {
  using ExecutionConfig = interpreter::ExecutionConfig;
  using InstructionSchedulingPriorityLess = std::function<bool(size_t, size_t)>;
  using SchedulingQueue =
      std::priority_queue<size_t,
                          std::vector<size_t>,
                          InstructionSchedulingPriorityLess>;

 public:
  PirInterpreter(const phi::Place& place,
                 const std::vector<std::string>& fetch_var_names,
                 const ::pir::Block* ir_block,
                 Scope* scope,
                 const ExecutionConfig& execution_config = ExecutionConfig());

  PirInterpreter(const phi::Place& place,
                 const std::vector<std::string>& fetch_var_names,
                 const ::pir::Block* ir_block,
                 Scope* scope,
                 std::shared_ptr<ValueExecutionInfo> value_exe_info,
                 const ExecutionConfig& execution_config = ExecutionConfig());

  ~PirInterpreter();

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false,
      bool switch_stream = false) override;

  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names,
                                   bool need_fetch = true,
                                   bool enable_job_schedule_profiler = false,
                                   bool enable_op_profiling = false,
                                   bool switch_stream = false) override;

  void ShareWorkQueueFrom(InterpreterBaseImpl* src) override;

  void ShareBuildResultsFrom(const InterpreterBaseImpl& src) override;

  std::tuple<double, double> InterpreterRunTime() override;

  std::shared_ptr<std::vector<size_t>> GetDependencyCount() const override;

  bool IsSharedResultsBuild() const override;

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog) override;

  std::shared_ptr<ProgramDesc> GetMutableCopyProgram() override;

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) override;

  const std::set<std::string>& JitInputVars() const override;

  void SetJitInputVars(const std::set<std::string>& jit_input_vars) override;

  const VariableScope* GetVariableScope() const override;

  void reset_scope(Scope* new_scope) override;

  const Scope* local_scope() const override;

  Scope* InnerScope() const;

  const phi::Place& GetPlace() const override { return place_; }

  void SetOutputHooks(const std::vector<HookFunc>& hookfuncs) override {}

  void SetInputHooks(const std::vector<HookFunc>& hookfuncs) override {}

  void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs) override {
    pir_output_hookfuncs_ = hookfuncs;
  }

  void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs) override {
    pir_input_hookfuncs_ = hookfuncs;
  }

  std::string GetNameByValue(::pir::Value value) const;

  // Only for debug
  Variable* DebugVar(const std::string& name) const override;

  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
  GetForceEventsToWaitInfo() {
    return force_events_to_wait_;
  }

  void SetForceEventsToWaitInfo(
      std::unordered_map<std::string, std::shared_ptr<EventInter>>*
          force_events_to_wait) {
    force_events_to_wait_ = force_events_to_wait;
  }

 private:
  // build graph
  void UpdateSyncOpNum();
  void UpdateNcclOpNum();
  void UpdateOneDNNOpNum();

  void AnalyseExecuteOrderForTrace(
      std::map<size_t, std::set<size_t>> op_downstream_map,
      InstructionSchedulingPriorityLess compare);
  void AnalyzeForceSyncOps();
  void ConstructEventForJitInput();
  void CalculateLastLiveOps();

  // gc
  void ClearDenseTensorArrayInLocalScope();

  // cuda graph
  void CheckCUDAGraphBeforeRun(const std::vector<std::string>& feed_names);
  void PrepareForCUDAGraphCapture();

  void Build(const std::vector<std::string>& feed_names,
             std::vector<paddle::framework::OpFuncNode>* op_func_nodes,
             bool switch_stream = false) override;

  bool IsStaticBuild() const override { return static_build_; }

  // workqueue
  std::shared_ptr<interpreter::AsyncWorkQueue> GetWorkQueue();

  // scope
  bool HasLocalScope() const;

  // For log and debug
  std::string GetDepsString() const;

  bool is_build_{false};
  bool static_build_{false};

  // Note(sonder): share the op dependency and event analysis procedure.
  bool is_shared_results_build_{false};

  const phi::Place place_;

  // from variable scope

  std::atomic<size_t> unfinished_op_number_{0};

  ExecutionConfig execution_config_;

  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
      force_events_to_wait_;

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

  // (*dependency_count_)[i] contains the number of dependencies that the i-th
  // op need to wait
  std::shared_ptr<std::vector<size_t>> dependency_count_;

  std::vector<std::shared_ptr<interpreter::OpDepInfo>> deps_;
  std::vector<std::shared_ptr<interpreter::VarRefInfo>> refs_;

  // used for Trace
  int64_t sync_op_num_{-1};
  int64_t nccl_op_num_{-1};
  int64_t onednn_op_num_{-1};
  std::vector<size_t> trace_execute_order_;

  std::vector<PirHookFunc> pir_output_hookfuncs_;
  std::vector<PirHookFunc> pir_input_hookfuncs_;

  /// ======================== ///
  ///        For new ir        ///
  /// ======================== ///
  std::string DebugValueInfo();

  std::string DebugInstructions();

  std::string DebugDependency();

  std::vector<std::string> DebugInfo();

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

  void SolvePersistableVarNames();

  const interpreter::PirDependencyBuilder& GetPirDependencyBuilder() const;

  const interpreter::PirStreamAnalyzer& GetPirStreamAnalyzer() const;

  InstructionSchedulingPriorityLess ir_instruction_scheduling_priority_less;

  const ::pir::Block* ir_block_{nullptr};

  std::unordered_map<::pir::Block*, PirInterpreter*> sub_blocks_;  // Not owned

  std::vector<std::unique_ptr<InstructionBase>> vec_instruction_base_;

  // value execution info
  std::shared_ptr<ValueExecutionInfo> value_exe_info_;

  std::vector<int> var_ref_count_;

  interpreter::PirDependencyBuilder ir_dependency_builder_;

  interpreter::PirStreamAnalyzer ir_stream_analyzer_;

  std::vector<std::string> fetch_var_names_;

  // Note(zhangbo): set_parameter_op's input and parameter_op's output
  // belongs to a parameter and cannot GC.
  std::unordered_set<std::string> parameter_var_names_;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::unique_ptr<phi::CalculateStreamTimer> calculate_stream_timer_;
#endif
  size_t last_calculate_instr_id_;
  bool enable_job_schedule_profiler_;
};

}  // namespace framework
}  // namespace paddle
