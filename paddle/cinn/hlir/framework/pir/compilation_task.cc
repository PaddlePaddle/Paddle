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
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace framework {

void GroupCompilationContext::SetLoweredFuncs(
    BucketLoweredFuncsWrapper&& funcs) {
  for (std::tuple<ir::SymbolicPredicate, ir::LoweredFunc, int>& predicate2func :
       funcs.predicate2funcs) {
    bucket_predicates_.push_back(std::move(std::get<0>(predicate2func)));
    lowered_funcs_.push_back(std::move(std::get<1>(predicate2func)));
    priorities_.push_back(std::move(std::get<2>(predicate2func)));
  }
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs.predicate2funcsCX86) {
    CX86_bucket_predicates_.push_back(std::move(predicate2func.first));
    CX86_lowered_funcs_.push_back(std::move(predicate2func.second));
  }
  infer_shape_lowered_funcs_.push_back(std::move(funcs.infer_shape_func));
}

std::string GroupCompilationContext::PrintPredicate2Funcs() const {
  std::stringstream ss;
  for (int i = 0; i < bucket_predicates_.size(); ++i) {
    ss << "[CONDITION " << i << "]: " << bucket_predicates_[i] << "\n";
    ss << "[LOWEREDFUNC " << i << "]: " << lowered_funcs_[i] << "\n\n";
  }
  return ss.str();
}

template <typename T>
void AppendVector(std::vector<T>* first, const std::vector<T>& second) {
  first->insert(first->end(), second.begin(), second.end());
}

GroupCompilationContext ContextReduction(
    std::vector<GroupCompilationContext>* contexts) {
  CHECK_GT(contexts->size(), 0);
  std::unordered_map<ir::Var, pir::CINNKernelInfo::ArgDimIdx> new_args_map;
  int total_args_num = 0;
  for (int i = 0; i < contexts->size(); ++i) {
    auto& func_args = (*contexts)[i].lowered_funcs_[0]->args;
    const auto& origin_int_args = (*contexts)[i].group_->int_args_map();
    std::vector<ir::Argument> new_args_vec;
    for (int arg_idx = 0; arg_idx < func_args.size(); ++arg_idx) {
      if (func_args[arg_idx].is_var()) {
        new_args_map[func_args[arg_idx].var_arg()] =
            origin_int_args.at(arg_idx);
      } else {
        new_args_vec.emplace_back(func_args[arg_idx]);
      }
    }
    for (auto& func : (*contexts)[i].lowered_funcs_) {
      func->args = new_args_vec;
    }
    if (i == 0) {
      total_args_num = new_args_vec.size();
    }
  }
  (*contexts)[0].group_->mut_int_args_map().clear();
  std::vector<ir::Argument> new_int_args_vec;
  for (const auto& [arg, idx_info] : new_args_map) {
    (*contexts)[0].group_->mut_int_args_map()[total_args_num++] = idx_info;
    new_int_args_vec.emplace_back(ir::Argument{arg});
  }
  for (int i = 0; i < contexts->size(); ++i) {
    for (auto& func : (*contexts)[i].lowered_funcs_) {
      AppendVector(&(func->args), new_int_args_vec);
    }
  }
  GroupCompilationContext result = (*contexts)[0];
  const auto GetExtendBroadcastPredicates =
      [](const GroupCompilationContext& ctx) -> std::vector<ir::Expr> {
    CHECK_EQ(ctx.broadcast_predicates_.size(), 1);
    int extend_size = ctx.bucket_predicates_.size();
    return std::vector<ir::Expr>(extend_size, ctx.broadcast_predicates_[0]);
  };
  result.broadcast_predicates_ = GetExtendBroadcastPredicates((*contexts)[0]);
  for (int i = 1; i < contexts->size(); ++i) {
    AppendVector(&result.broadcast_predicates_,
                 GetExtendBroadcastPredicates((*contexts)[i]));
    AppendVector(&result.bucket_predicates_, (*contexts)[i].bucket_predicates_);
    AppendVector(&result.priorities_, (*contexts)[i].priorities_);
    AppendVector(&result.lowered_funcs_, (*contexts)[i].lowered_funcs_);
    AppendVector(&result.CX86_bucket_predicates_,
                 (*contexts)[i].CX86_bucket_predicates_);
    AppendVector(&result.CX86_lowered_funcs_,
                 (*contexts)[i].CX86_lowered_funcs_);
    AppendVector(&result.infer_shape_lowered_funcs_,
                 (*contexts)[i].infer_shape_lowered_funcs_);
  }
  return result;
}

void LoweringTask::operator()() { Lowering(); }

void LoweringTask::Lowering() {
  VLOG(5) << "Begin to lowering group: " << *context_->group_;
  auto op_lowerer = CreateOpLowerer<pir::OpLoweringGroupPtr>(context_->target_);
  context_->SetLoweredFuncs(
      op_lowerer.BucketLower(context_->group_,
                             /* apply op schedule = */ false,
                             /* apply group schedule = */ true,
                             /* apply pass = */ true));
  if (context_->group_->IsBroadcastLeaf()) {
    const auto& broadcast_conditions =
        context_->group_->GetBroadcastConditions();
    const auto ChangeEqualDimExprToExpr =
        [&broadcast_conditions]() -> ir::Expr {
      ir::Expr result = ir::Expr(true);
      using BranchType = pir::OpLoweringGroup::BranchType;
      for (const auto& [broadcast_expr, condition] : broadcast_conditions) {
        const auto& expr_converter = common::DimExprConverter();
        ir::Expr lhs = expr_converter.ConvertToIrExpr(broadcast_expr->lhs);
        ir::Expr rhs = expr_converter.ConvertToIrExpr(broadcast_expr->rhs);
        ir::Expr one = ir::Expr(1);
        ir::Expr condition_expr;
        if (condition == BranchType::LHS_EQ_RHS) {
          condition_expr = ir::EQ::Make(lhs, rhs);
        } else {
          condition_expr = ir::NE::Make(lhs, rhs);
          ir::Expr eq_one_expr;
          if (condition == BranchType::LHS_EQ_ONE) {
            eq_one_expr = ir::EQ::Make(lhs, one);
          } else {  // BranchType::RHS_EQ_ONE
            eq_one_expr = ir::EQ::Make(rhs, one);
          }
          condition_expr = ir::And::Make(condition_expr, eq_one_expr);
        }
        result = ir::And::Make(result, condition_expr);
      }
      return result;
    };
    context_->broadcast_predicates_.emplace_back(ChangeEqualDimExprToExpr());
  }
  VLOG(5) << "End to lowering: " << context_->PrintPredicate2Funcs();
}

std::shared_ptr<pir::CompilationResult> CompilationTask::operator()() {
  return CodegenAndJit();
}

std::shared_ptr<pir::CompilationResult> CompilationTask::CodegenAndJit() {
  ir::Module::Builder builder(cinn::common::UniqName("module"),
                              context_->target_);
  PADDLE_ENFORCE_EQ(context_->bucket_predicates_.size(),
                    context_->lowered_funcs_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));
  PADDLE_ENFORCE_EQ(context_->bucket_predicates_.size(),
                    context_->priorities_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and priorites should be "
                        "the same."));
  for (const ir::Expr& predicate : context_->bucket_predicates_) {
    builder.AddBucketPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->lowered_funcs_) {
    builder.AddFunction(func);
  }
  for (int& priority : context_->priorities_) {
    builder.AddPriority(priority);
  }
  for (const ir::Expr& func : context_->infer_shape_lowered_funcs_) {
    builder.AddInferShapeFunc(func);
  }
  ir::Module::Builder builder_CX86(cinn::common::UniqName("module"),
                                   common::DefaultHostTarget());

  PADDLE_ENFORCE_EQ(context_->CX86_bucket_predicates_.size(),
                    context_->CX86_lowered_funcs_.size(),
                    phi::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));
  for (const ir::Expr& predicate : context_->CX86_bucket_predicates_) {
    builder_CX86.AddBucketPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : context_->CX86_lowered_funcs_) {
    builder_CX86.AddFunction(func);
  }
  if (context_->broadcast_predicates_.size() > 0) {
    for (const ir::Expr& predicate : context_->broadcast_predicates_) {
      builder.AddBroadcastPredicate(predicate);
      builder_CX86.AddBroadcastPredicate(predicate);
    }
  }
  ir::Module ir_module = builder.Build();
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
