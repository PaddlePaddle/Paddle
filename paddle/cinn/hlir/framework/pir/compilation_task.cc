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
  if (CompilationCache::Instance().Has(context_->group)) {
    VLOG(4) << "Found cached kernel info for group: "
            << context_->group_->FuncName();
    return;
  }
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
  BuildPirCINNKernelInfo(ir_module)
}

pir::CINNKernelInfo CompilationTask::GetCINNKernelInfo() {
  if (!CompilationCache::Instance().Has(context_->group)) {
    PADDLE_THROW(phi::errors::NotFound(
        "Kernel info has been cached for current group."));
  }
  return CompilationCache::Instance().Get(context_->group)->GetKernelInfo();
}

void CompilationTask::BuildPirCINNKernelInfo(const ir::Module& module) {
  auto compilation_result =
      std::make_shared<CompilationResult>(context_->target_);
  BackendResource& backend_resource = compilation_result->GetBackendResource();
  backend_resource->backend_compiler_->Build(ir_module, "");
  backend_resource->host_fn_name_ = context_->group_->FuncName();
  backend_resource->infer_fn_name_ =
      backend_resource->host_fn_name + "_infer_shape";
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
