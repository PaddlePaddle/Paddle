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

#include "paddle/fluid/framework/new_executor/interpretercore.h"

#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/framework/new_executor/program_interpreter.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

PHI_DEFINE_EXPORTED_bool(
    new_executor_serial_run,
    false,
    "Enable serial execution for standalone executor, used for debug.");
PHI_DEFINE_EXPORTED_bool(
    new_executor_static_build,
    false,
    "Build the interpreterCore statically without running kernels.");
PHI_DEFINE_EXPORTED_bool(new_executor_use_inplace,
                         false,
                         "Use inplace in new executor");
PHI_DEFINE_EXPORTED_bool(new_executor_use_local_scope,
                         true,
                         "Use local_scope in new executor(especially used "
                         "in UT), can turn off for better performance");

namespace paddle::framework {

InterpreterCore::InterpreterCore(const phi::Place& place,
                                 const BlockDesc& block,
                                 framework::Scope* scope,
                                 const ExecutionConfig& execution_config)
    : impl_(nullptr), fetch_var_names_() {
  VLOG(4) << "InterpreterCore(): " << this << " on " << place;
  impl_ = std::make_unique<ProgramInterpreter>(
      place, block, scope, execution_config);
}

InterpreterCore::InterpreterCore(
    const phi::Place& place,
    const std::vector<std::string>& fetch_var_names,
    const ::pir::Block* ir_block,
    framework::Scope* scope,
    const ExecutionConfig& execution_config)
    : impl_(nullptr), fetch_var_names_() {
  VLOG(4) << "InterpreterCore(): " << this << " on " << place;
  impl_ = std::make_unique<PirInterpreter>(
      place, fetch_var_names, ir_block, scope, execution_config);
}

InterpreterCore::~InterpreterCore() {
  VLOG(4) << "~InterpreterCore(): " << this;
  impl_.reset(nullptr);
}

FetchList InterpreterCore::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors,
    bool need_fetch,
    bool enable_job_schedule_profiler,
    bool switch_stream) {
  return impl_->Run(feed_names,
                    feed_tensors,
                    need_fetch,
                    enable_job_schedule_profiler,
                    switch_stream);
}

FetchList InterpreterCore::Run(const std::vector<std::string>& feed_names,
                               bool need_fetch,
                               bool enable_job_schedule_profiler,
                               bool enable_op_profiling,
                               bool switch_stream) {
  return impl_->Run(feed_names,
                    need_fetch,
                    enable_job_schedule_profiler,
                    enable_op_profiling,
                    switch_stream);
}

void InterpreterCore::ShareWorkQueueFrom(std::shared_ptr<InterpreterCore> src) {
  impl_->ShareWorkQueueFrom(const_cast<InterpreterBaseImpl*>(src->Impl()));
}

void InterpreterCore::ShareBuildResultsFrom(
    std::shared_ptr<InterpreterCore> src) {
  // ShareBuildResultsFrom required const InterpreterBaseImpl& src as input
  impl_->ShareBuildResultsFrom(*src->Impl());
}

void InterpreterCore::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  impl_->SetCopyProgram(prog);
}

void InterpreterCore::SetSkipGcVars(const std::set<std::string>& skip_gc_vars) {
  impl_->SetSkipGcVars(skip_gc_vars);
}

const std::set<std::string>& InterpreterCore::JitInputVars() const {
  return impl_->JitInputVars();
}

void InterpreterCore::SetJitInputVars(
    const std::set<std::string>& jit_input_vars) {
  impl_->SetJitInputVars(jit_input_vars);
}

const VariableScope* InterpreterCore::GetVariableScope() const {
  return impl_->GetVariableScope();
}

void InterpreterCore::reset_scope(Scope* new_scope) {
  impl_->reset_scope(new_scope);
}

const Scope* InterpreterCore::local_scope() const {
  return impl_->local_scope();
}

const phi::Place& InterpreterCore::GetPlace() const {
  return impl_->GetPlace();
}

void InterpreterCore::SetInputHooks(const std::vector<HookFunc>& hookfuncs) {
  impl_->SetInputHooks(hookfuncs);
}

void InterpreterCore::SetOutputHooks(const std::vector<HookFunc>& hookfuncs) {
  impl_->SetOutputHooks(hookfuncs);
}

void InterpreterCore::SetInputHooks(const std::vector<PirHookFunc>& hookfuncs) {
  impl_->SetInputHooks(hookfuncs);
}

void InterpreterCore::SetOutputHooks(
    const std::vector<PirHookFunc>& hookfuncs) {
  impl_->SetOutputHooks(hookfuncs);
}

void InterpreterCore::Build(
    const std::vector<std::string>& feed_names,
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  impl_->Build(feed_names, op_func_nodes);
}

bool InterpreterCore::IsStaticBuild() const { return impl_->IsStaticBuild(); }

std::tuple<double, double> InterpreterCore::InterpreterRunTime() {
  return impl_->InterpreterRunTime();
}

std::shared_ptr<ProgramDesc> InterpreterCore::GetMutableCopyProgram() {
  return impl_->GetMutableCopyProgram();
}

Variable* InterpreterCore::DebugVar(const std::string& name) const {
  return impl_->DebugVar(name);
}

}  // namespace paddle::framework
