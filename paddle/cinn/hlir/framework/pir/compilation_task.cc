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
#include "paddle/common/enforce.h"

namespace cinn {
namespace hlir {
namespace framework {

void GroupCompilationContext::SetLoweredFuncs(
    BucketLoweredFuncsWrapper&& funcs) {
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs.predicate2funcs) {
    predicates_.push_back(std::move(predicate2func.first));
    lowered_funcs_.push_back(std::move(predicate2func.second));
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

void CompilationTask::operator()() {
  VLOG(4) << "Run Compilation Task for : " << context_->group_.get();
  if (CompilationCache::Instance().Has(context_->group_)) {
    VLOG(4) << "Found cached kernel info for group: "
            << context_->group_->FuncName();
    return;
  }
  Lowering();
  CodegenAndJit();
}

void CompilationTask::Lowering() {
  auto op_lowerer = CreateOpLowerer<pir::OpLoweringGroupPtr>(context_->target_);
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
  BuildPirCINNKernelInfo(ir_module);
}

pir::CINNKernelInfo CompilationTask::GetCINNKernelInfo() {
  if (!CompilationCache::Instance().Has(context_->group_)) {
    PADDLE_THROW(phi::errors::NotFound(
        "Kernel info has been cached for current group."));
  }
  return CompilationCache::Instance().GetKernelInfo(context_->group_);
}

void CompilationTask::BuildPirCINNKernelInfo(const ir::Module& module) {
  auto compilation_result =
      std::make_shared<pir::CompilationResult>(context_->target_);
  pir::BackendResource& backend_resource =
      compilation_result->MutableBackendResource();
  backend_resource.GetBackendCompiler()->Build(module, "");
  backend_resource.SetHostFnName(context_->group_->FuncName());
  backend_resource.SetInferFnName(context_->group_->FuncName() +
                                  "_infer_shape");
  CompilationCache::Instance().Insert(context_->group_, compilation_result);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
