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
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace hlir {
namespace framework {

void GroupCompilationContext::SetLoweredFuncs(
    BucketLoweredFuncsWrapper&& funcs) {
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs.predicate2funcs) {
    predicates_.push_back(std::move(predicate2func.first));
    lowered_funcs_.push_back(std::move(predicate2func.second));
    ++func_size_;
  }
  infer_shape_lowered_func_ = std::move(funcs.infer_shape_func);
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
      op_lowerer.BucketLower(context_->group_,
                             /* apply op schedule = */ false,
                             /* apply group schedule = */ true,
                             /* apply pass = */ true));
}

void CompilationTask::CodegenAndJit() {
  ir::Module::Builder builder(cinn::common::UniqName("module"),
                              context_->target_);
  CHECK_EQ(context_->predicates_.size(), context_->lowered_funcs_.size());
  for (const ir::Expr& predicate : context_->predicates_) {
    builder.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->lowered_funcs_) {
    builder.AddFunction(func);
  }
  builder.SetInferShapeFunc(context_->infer_shape_lowered_func_);
  ir::Module ir_module = builder.Build();

  context_->backend_compiler_ = backends::Compiler::Create(context_->target_);
  context_->backend_compiler_->Build(ir_module, "");
}

pir::CINNKernelInfo CompilationTask::BuildPirCINNKernelInfo() {
  std::string fn_name = context_->group_->FuncName();
  VLOG(4) << "Lookup kernel name: " << fn_name;
  auto* fn_ptr = context_->backend_compiler_->Lookup(fn_name);
  CHECK(fn_ptr);
  auto* infer_shape_fn_ptr =
      context_->backend_compiler_->Lookup(fn_name + "_infer_shape");
  CHECK(infer_shape_fn_ptr);
  pir::CINNKernelInfo cinn_kernel_info;
  cinn_kernel_info.fn_name = fn_name;
  cinn_kernel_info.fn_ptr = fn_ptr;
  cinn_kernel_info.infer_shape_fn_ptr = infer_shape_fn_ptr;
  cinn_kernel_info.int_args_map = context_->group_->int_args_map;
  return cinn_kernel_info;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
