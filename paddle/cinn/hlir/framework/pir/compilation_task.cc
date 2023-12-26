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

#include "paddle/cinn/hlir/framework/pir/compilation_task.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace hlir {
namespace framework {

void GroupCompilationContext::SetLoweredFuncs(
    std::vector<std::pair<ir::SymbolicPredicate, ir::LoweredFunc>>&& funcs) {
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs) {
    predicates_.push_back(predicate2func.first);
    lowered_funcs_.push_back(predicate2func.second);
    ++func_size_;
  }
}

std::string GroupCompilationContext::PrintPredicate2Funcs() const {
  std::stringstream ss;
  for (int i = 0; i < predicates_.size(); ++i) {
    ss << "[CONDITION " << i << "]: " << predicates_[i] << "\n";
    ss << "[LOWEREDFUNC " << i << "]: " << lowered_funcs_[i] << "\n\n";
  }
  return ss.str();
}

void* GroupCompilationContext::FuncPtr() {
  return backend_compiler_->Lookup(host_func_name_);
}

std::shared_ptr<backends::Compiler> GroupCompilationContext::BackendCompiler() {
  return backend_compiler_;
}

void CompilationTask::operator()() {
  Lowering();
  CodegenAndJit();
}

void CompilationTask::Lowering() {
  auto op_lowerer = CreateOpLowerer<pir::GroupPtr>(context_->target_);
  context_->SetLoweredFuncs(
      op_lowerer.BucketLower(context_->group_, false, true, false));
  // context_->SetLoweredFuncs(
  //     op_lowerer.BucketLower(context_->group_, false, false, false));
  op_lowerer.InsertNameGeneToScope(context_->scope_);
}

void CompilationTask::CodegenAndJit() {
  ir::Module::Builder builder(cinn::common::UniqName("module"),
                              context_->target_);
  CHECK_EQ(context_->predicates_.size(), context_->lowered_funcs_.size());
  for (const ir::Expr predicate : context_->predicates_) {
    builder.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->lowered_funcs_) {
    builder.AddFunction(func);
  }
  ir::Module ir_module = builder.Build();

  context_->backend_compiler_ = backends::Compiler::Create(context_->target_);
  context_->backend_compiler_->Build(ir_module, "");
}

std::unique_ptr<Instruction> CompilationTask::BuildInstruction() {
  std::string fn_name = context_->group_->FuncName();
  std::unique_ptr<Instruction> instr =
      std::make_unique<Instruction>(context_->target_,
                                    context_->scope_.get(),
                                    context_->group_->input_names,
                                    context_->group_->output_names,
                                    fn_name);
  VLOG(4) << "Lookup kernel name: " << fn_name;
  auto* fn_ptr = context_->backend_compiler_->Lookup(fn_name);
  CHECK(fn_ptr);
  instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), fn_name);
  instr->Finalize();
  return instr;
}

pir::CINNKernelInfo CompilationTask::BuildPirCINNKernelInfo() {
  std::string fn_name = context_->group_->FuncName();
  VLOG(4) << "Lookup kernel name: " << fn_name;
  auto* fn_ptr = context_->backend_compiler_->Lookup(fn_name);
  CHECK(fn_ptr);
  pir::CINNKernelInfo cinn_kernel_info;
  cinn_kernel_info.fn_ptr = fn_ptr;
  cinn_kernel_info.int_args_map = context_->group_->int_args_map;
  return cinn_kernel_info;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
