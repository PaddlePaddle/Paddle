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

#include "paddle/fluid/framework/new_executor/interpreter_base_impl.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

PD_DECLARE_bool(new_executor_use_local_scope);

namespace pir {
class Block;
}  // namespace pir

namespace paddle {
namespace framework {

class InterpreterBaseImpl;

class InterpreterCore {
  using ExecutionConfig = interpreter::ExecutionConfig;
  using HookFunc = std::function<void(OperatorBase*, Scope*)>;

 public:
  InterpreterCore(const phi::Place& place,
                  const BlockDesc& block,
                  Scope* scope,
                  const ExecutionConfig& execution_config = ExecutionConfig());
  // This constructor is for New IR.
  TEST_API InterpreterCore(
      const phi::Place& place,
      const std::vector<std::string>& fetch_var_names,
      const ::pir::Block* ir_prog,
      Scope* scope,
      const ExecutionConfig& execution_config = ExecutionConfig());
  TEST_API ~InterpreterCore();

  const InterpreterBaseImpl* Impl() const { return impl_.get(); }

  TEST_API paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false,
      bool switch_stream = false);

  TEST_API paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false,
      bool enable_op_profiling = false,
      bool switch_stream = false);

  void RunProfile(const std::vector<std::string>& feed_names);

  std::shared_ptr<ProgramDesc> GetMutableCopyProgram();

  void ShareWorkQueueFrom(std::shared_ptr<InterpreterCore> src);

  void ShareBuildResultsFrom(std::shared_ptr<InterpreterCore> src);

  void SetCopyProgram(std::shared_ptr<ProgramDesc> prog);

  TEST_API void SetSkipGcVars(const std::set<std::string>& skip_gc_vars);

  const std::set<std::string>& JitInputVars() const;

  void SetJitInputVars(const std::set<std::string>& jit_input_vars);

  const VariableScope* GetVariableScope() const;

  void reset_scope(Scope* new_scope);

  const Scope* local_scope() const;

  const phi::Place& GetPlace() const;

  void SetOutputHooks(const std::vector<HookFunc>& hookfuncs);

  void SetInputHooks(const std::vector<HookFunc>& hookfuncs);

  void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs);

  void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs);

  void Build(const std::vector<std::string>& feed_names,
             std::vector<paddle::framework::OpFuncNode>* op_func_nodes);

  bool IsStaticBuild() const;

  std::tuple<double, double> InterpreterRunTime();

  // Only for debug
  TEST_API Variable* DebugVar(const std::string& name) const;

 private:
  DISABLE_COPY_AND_ASSIGN(InterpreterCore);

  std::unique_ptr<InterpreterBaseImpl> impl_;

  std::vector<std::string> fetch_var_names_;
};

}  // namespace framework
}  // namespace paddle
