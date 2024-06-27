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
  for (std::tuple<ir::SymbolicPredicate, ir::LoweredFunc, int>& predicate2func :
       funcs.predicate2funcs) {
    predicates_.push_back(std::move(std::get<0>(predicate2func)));
    lowered_funcs_.push_back(std::move(std::get<1>(predicate2func)));
    priorities_.push_back(std::move(std::get<2>(predicate2func)));
  }
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs.predicate2funcsCX86) {
    CX86_predicates_.push_back(std::move(predicate2func.first));
    CX86_lowered_funcs_.push_back(std::move(predicate2func.second));
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

std::shared_ptr<pir::CompilationResult> CompilationTask::operator()() {
  Lowering();
  return CodegenAndJit();
}

void CompilationTask::Lowering() {
  VLOG(5) << "Begin to lowering group: " << *context_->group_;
  auto op_lowerer = CreateOpLowerer<pir::OpLoweringGroupPtr>(context_->target_);
  context_->SetLoweredFuncs(
      op_lowerer.BucketLower(context_->group_,
                             /* apply op schedule = */ false,
                             /* apply group schedule = */ true,
                             /* apply pass = */ true));
  VLOG(5) << "End to lowering: " << context_->PrintPredicate2Funcs();
}

std::shared_ptr<pir::CompilationResult> CompilationTask::CodegenAndJit() {
  ir::Module::Builder builder(cinn::common::UniqName("module"),
                              context_->target_);
  PADDLE_ENFORCE_EQ(context_->predicates_.size(),
                    context_->lowered_funcs_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));
  PADDLE_ENFORCE_EQ(context_->predicates_.size(),
                    context_->priorities_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and priorites should be "
                        "the same."));
  for (const ir::Expr& predicate : context_->predicates_) {
    builder.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->lowered_funcs_) {
    builder.AddFunction(func);
  }
  for (int& priority : context_->priorities_) {
    builder.AddPriority(priority);
  }
  builder.SetInferShapeFunc(context_->infer_shape_lowered_func_);
  ir::Module ir_module = builder.Build();

  ir::Module::Builder builder_CX86(cinn::common::UniqName("module"),
                                   common::DefaultHostTarget());
  PADDLE_ENFORCE_EQ(context_->CX86_predicates_.size(),
                    context_->CX86_lowered_funcs_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));
  for (const ir::Expr& predicate : context_->CX86_predicates_) {
    builder_CX86.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->CX86_lowered_funcs_) {
    builder_CX86.AddFunction(func);
  }
  ir::Module ir_moduleCX86 = builder_CX86.Build();

  return BuildPirCINNKernelInfo(ir_module, ir_moduleCX86);
}

std::shared_ptr<pir::CompilationResult> CompilationTask::BuildPirCINNKernelInfo(
    const ir::Module& module, const ir::Module& CX86module) {
  auto compilation_result =
      std::make_shared<pir::CompilationResult>(context_->target_);
  auto backend_resource = std::make_shared<pir::BackendResource>(
      context_->target_,
      context_->group_->FuncName(),
      context_->group_->FuncName() + "_infer_shape",
      context_->group_->int_args_map());
  VLOG(5) << "Start to compile module into cuda kernel...";
  backend_resource->GetBackendCompiler()->Build(module, "", false);
  backend_resource->GetBackendCompiler()->AppendCX86(CX86module);
  compilation_result->SetBackendResource(backend_resource);
  VLOG(5) << "End to compile module into cuda kernel.";
  return compilation_result;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
