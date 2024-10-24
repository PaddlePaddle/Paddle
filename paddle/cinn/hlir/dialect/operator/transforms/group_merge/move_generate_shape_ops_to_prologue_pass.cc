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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/move_generate_shape_ops_to_prologue_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

namespace cinn {
namespace dialect {

namespace {

class GroupOpGenerateShapeOpsPattern
    : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  explicit GroupOpGenerateShapeOpsPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    pir::ShapeConstraintIRAnalysis& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    ShapeOrDataDimExprsAccessor dim_exprs_accessor{
        .GetShapeOrDataDimExprs =
            [&](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
          return shape_analysis.GetShapeOrDataForValue(value);
        }};
    return MoveGenerateShapeOpsToPrologue(
        ctx, group_op.block(), dim_exprs_accessor);
  }
};

class MoveGenerateShapeOpsToProloguePass : public pir::Pass {
 public:
  MoveGenerateShapeOpsToProloguePass()
      : pir::Pass("move_generate_shape_ops_to_prologue", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto group_op = op->dyn_cast<cinn::dialect::GroupOp>();
    PADDLE_ENFORCE(
        group_op,
        ::common::errors::InvalidArgument("The operation for group is failed"));
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    pir::ShapeConstraintIRAnalysis& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    ShapeOrDataDimExprsAccessor dim_exprs_accessor{
        .GetShapeOrDataDimExprs =
            [&](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
          return shape_analysis.GetShapeOrDataForValue(value);
        }};
    MoveGenerateShapeOpsToPrologue(ctx, group_op.block(), dim_exprs_accessor);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<cinn::dialect::GroupOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace ir {

std::unique_ptr<::pir::Pass> CreateMoveGenerateShapeOpsToProloguePass() {
  return std::make_unique<MoveGenerateShapeOpsToProloguePass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, CinnGroupLoweringPass);
