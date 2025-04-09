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

#include "paddle/fluid/framework/new_executor/interpreter_base_impl.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#endif

namespace paddle {
namespace framework {

///
/// \brief Derived Class to interpret the instructions transformed
/// from legacy ProgramDesc.
///

class ProgramInterpreter : public InterpreterBaseImpl {
  using ExecutionConfig = interpreter::ExecutionConfig;
  using InstructionSchedulingPriorityLess = std::function<bool(size_t, size_t)>;
  using SchedulingQueue =
      std::priority_queue<size_t,
                          std::vector<size_t>,
                          InstructionSchedulingPriorityLess>;

 public:
  ProgramInterpreter(
      const phi::Place& place,
      const BlockDesc& block,
      Scope* scope,
      const ExecutionConfig& execution_config = ExecutionConfig());

  ~ProgramInterpreter();

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

  std::shared_ptr<ProgramDesc> GetMutableCopyProgram() override;

  void Build(const std::vector<std::string>& feed_names,
             std::vector<paddle::framework::OpFuncNode>* op_func_nodes,
             bool switch_stream = false) override;

  void ShareWorkQueueFrom(InterpreterBaseImpl* src) override;

  void ShareBuildResultsFrom(const InterpreterBaseImpl& src) override;

  // op dependences
  const interpreter::DependencyBuilder& GetDependencyBuilder() const;

  std::shared_ptr<std::vector<size_t>> GetDependencyCount() const override;

  const interpreter::StreamAnalyzer& GetStreamAnalyzer() const;

  bool IsSharedResultsBuild() const override;

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog) override;

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) override;

  const std::set<std::string>& JitInputVars() const override;

  void SetJitInputVars(const std::set<std::string>& jit_input_vars) override;

  const VariableScope* GetVariableScope() const override;

  void reset_scope(Scope* new_scope) override;

  const Scope* local_scope() const override;

  const phi::Place& GetPlace() const override { return place_; }

  void SetOutputHooks(const std::vector<HookFunc>& hookfuncs) override {
    output_hookfuncs_ = hookfuncs;
  }

  void SetInputHooks(const std::vector<HookFunc>& hookfuncs) override {
    input_hookfuncs_ = hookfuncs;
  }

  void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs) override {}

  void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs) override {}

  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
  GetForceEventsToWaitInfo() {
    return force_events_to_wait_;
  }

  void SetForceEventsToWaitInfo(
      std::unordered_map<std::string, std::shared_ptr<EventInter>>*
          force_events_to_wait) {
    force_events_to_wait_ = force_events_to_wait;
  }

  bool IsStaticBuild() const override { return static_build_; }

  std::tuple<double, double> InterpreterRunTime() override;

  // Only for debug
  Variable* DebugVar(const std::string& name) const override;

 private:
  // build graph
  void Convert(std::vector<paddle::framework::OpFuncNode>* op_func_nodes);
  void BuildOperatorDependences();
  void BuildAndCacheInstructionCtx(Instruction* instr_node);
  void BuildSkipShareLoDInfo();
  void UpdateSyncOpNum();
  void AnalyseExecuteOrderForTrace();

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
               bool prepare_feed,
               bool switch_stream = false);

  void RecordMemcpyD2H(const Instruction& instr_node);

  // gc
  void RecordStreamForGC(const Instruction& instr);
  void CheckGC(const Instruction& instr);
  void ClearDenseTensorArrayInLocalScope();

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

  // op profiling status
  bool is_in_op_profiling_mode_{false};

  const phi::Place place_;
  const BlockDesc& block_;  // not owned

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

  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
      force_events_to_wait_;

  VariableScope var_scope_;
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
  std::vector<size_t> trace_execute_order_;

  InstructionSchedulingPriorityLess instruction_scheduling_priority_less;

  std::vector<HookFunc> output_hookfuncs_;
  std::vector<HookFunc> input_hookfuncs_;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::unique_ptr<phi::CalculateStreamTimer> calculate_stream_timer_;
#endif
  size_t last_calculate_instr_id_;
  bool enable_job_schedule_profiler_;
};

static inline const phi::DenseTensor& GetTensorFromVar(const Variable* var) {
  if (var->IsType<phi::DenseTensor>()) {
    return var->Get<phi::DenseTensor>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Variable must be type of phi::DenseTensor, but received %s.",
        framework::ToTypeName(var->Type())));
  }
}

static inline phi::DenseTensor* GetMutableTensorFromVar(Variable* var) {
  if (var->IsType<phi::DenseTensor>()) {
    return var->GetMutable<phi::DenseTensor>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Variable must be type of phi::DenseTensor, but received %s.",
        framework::ToTypeName(var->Type())));
  }
}

}  // namespace framework
}  // namespace paddle
