// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/cinn/operator_fusion/fusion_tracker/expr_utils.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/interpreter.h"

namespace cinn::fusion {

void RunRenameInstr(const std::shared_ptr<RenameInstr>& instr,
                    FusionInterpreter* interpreter) {
  interpreter->scope[instr->new_name_] =
      interpreter->scope[instr->origin_name_];
  interpreter->scope.erase(instr->origin_name_);
}

void RunCombineInstr(const std::shared_ptr<CombineInstr>& instr,
                     FusionInterpreter* interpreter) {
  interpreter->scope[instr->result_] = CombineScopeElement(
      interpreter->scope[instr->first_], interpreter->scope[instr->second_]);
}

void RunInitPatternInstr(const std::shared_ptr<InitPatternInstr>& instr,
                         FusionInterpreter* interpreter) {
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace_back((interpreter->lowered_expr)[instr->op_]);
  interpreter->scope[instr->result_] = new_pattern;
}

void RunTrivialInlineInstr(const std::shared_ptr<TrivialInlineInstr>& instr,
                           FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->upstream_]->fusion_ops.size(), 1);
  auto upstream_op = std::get<TrivialOp>(
      interpreter->scope[instr->upstream_]->fusion_ops.front());
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  for (auto downstream_op :
       interpreter->scope[instr->downstream_]->fusion_ops) {
    new_pattern->fusion_ops.emplace_back(
        cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(
            upstream_op, downstream_op));
  }
  interpreter->scope[instr->result_] = new_pattern;
}

void RunTmpTransformInstr(const std::shared_ptr<TmpTransformInstr>& instr,
                          FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->upstream_]->fusion_ops.size(), 1);
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->downstream_]->fusion_ops.size(),
                    1);
  auto upstream_op = std::get<ReduceOp>(
      interpreter->scope[instr->upstream_]->fusion_ops.front());
  auto downstream_op =
      interpreter->scope[instr->downstream_]->fusion_ops.front();
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace(
      cinn::hlir::framework::pir::trivial_fusion_detail::
          TransformReduceLoopRange(
              upstream_op, &downstream_op, instr->fake_reduce_iter_idx_););
  interpreter->scope[instr->result_] = new_pattern;
}

void RunTrivialLoopAlignInstr(
    const std::shared_ptr<TrivialLoopAlignInstr>& instr,
    FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->upstream_]->fusion_ops.size(), 1);
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->downstream_]->fusion_ops.size(),
                    1);
  auto upstream_op = std::get<ReduceOp>(
      interpreter->scope[instr->upstream_]->fusion_ops.front());
  auto downstream_op = std::get<TrivialOp>(
      interpreter->scope[instr->downstream_]->fusion_ops.front());
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace(
      cinn::hlir::framework::pir::trivial_fusion_detail::SinkTrivialLoopAlign(
          downstream_op, upstream_op, instr->fake_reduce_iter_idx_););
  interpreter->scope[instr->result_] = new_pattern;
}

void RunAnchorTransformAndReturnInstr(
    const std::shared_ptr<AnchorTransformAndReturnInstr>& instr,
    FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->target_]->fusion_ops.size(), 1);
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();

  std::function<ir::Expr(ir::Expr)> do_transform =
      [transform_route = instr->transform_route_](ir::Expr target) -> ir::Expr {
    for (auto transform : transform_route) {
      target = std::visit(ApplyTransform(target), transform);
    }
    return target;
  };

  auto exprs = do_transform(std::visit(
      FusionOp2Expr(), interpreter->scope[instr->target_]->fusion_ops.front()));
  interpreter.ret_expr.insert(
      interpreter.ret_expr.end(), exprs.begin(), exprs.end());
}

void RunReturnInstr(const std::shared_ptr<ReturnInstr>& instr,
                    FusionInterpreter* interpreter) {
  for (auto fusion_op : interpreter->scope[instr->target_]->fusion_ops) {
    auto exprs = std::visit(FusionOp2Expr(), fusion_op);
    interpreter->ret_expr.insert(
        interpreter->ret_expr.end(), exprs.begin(), exprs.end());
  }
}

std::vector<ir::Expr> FusionInterpreter::Run() {
  for (auto instr : tracker->instructions_) {
    switch (instr->type()) {
      case T_Rename:
        RunRenameInstr(dynamic_cast_instr_with_err<RenameInstr>(instr), this);
        break;
      case T_Combine:
        RunCombineInstr(dynamic_cast_instr_with_err<CombineInstr>(instr), this);
        break;
      case T_InitPattern:
        RunInitPatternInstr(
            dynamic_cast_instr_with_err<InitPatternInstr>(instr), this);
        break;
      case T_TrivialInline:
        RunTrivialInlineInstr(
            dynamic_cast_instr_with_err<TrivialInlineInstr>(instr), this);
        break;
      case T_TmpTransform:
        RunTmpTransformInstr(
            dynamic_cast_instr_with_err<TmpTransformInstr>(instr), this);
        break;
      case T_AnchorTransformAndReturn:
        RunAnchorTransformAndReturnInstr(
            dynamic_cast_instr_with_err<AnchorTransformAndReturnInstr>(instr),
            this);
        break;
      case T_Return:
        RunReturnInstr(dynamic_cast_instr_with_err<ReturnInstr>(instr), this);
      default:
        PADDLE_THROW("Unsupported Fusion Instrution");
    }
  }

  return TopoSort(ret_expr);
}

}  // namespace cinn::fusion
