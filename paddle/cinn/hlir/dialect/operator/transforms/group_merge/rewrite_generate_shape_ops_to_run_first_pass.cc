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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/cinn_group_lowering_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/rewrite_generate_shape_ops_to_run_first_pass.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"

namespace cinn {
namespace dialect {

namespace {

class GroupOpGenerateShapeOpsPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  GroupOpGenerateShapeOpsPattern(
      ::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    pir::ShapeConstraintIRAnalysis& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    ShapeOrDataDimExprsAccessor dim_exprs_accessor{
      .GetShapeOrDataDimExprs=[&](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
        return shape_analysis.value_id_to_shapeordata_.at(GetValueId(&value));
      },
      .SetShapeOrDataDimExprs=[&](pir::Value value, const symbol::ShapeOrDataDimExprs& dim_exprs) {
        shape_analysis.value_id_to_shapeordata_[GetValueId(&value)] = dim_exprs;
      }
    };
    return RewriteGenerateShapeOpToRunFirst(ctx, group_op.block(), dim_exprs_accessor);
  }

};

class RewriteGenerateShapeOpsToRunFirstPass : public pir::PatternRewritePass {
 public:
  RewriteGenerateShapeOpsToRunFirstPass()
      : pir::PatternRewritePass("generate_shape_ops_to_run_first", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<GroupOpGenerateShapeOpsPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    auto* program = op->GetParentProgram();
    VLOG(4) << "Before RewriteGenerateShapeOpsToRunFirstPass: " << *program;
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

};

}  // namespace

namespace ir {

std::unique_ptr<::pir::Pass> CreateRewriteGenerateShapeOpsToRunFirstPass() {
  return std::make_unique<RewriteGenerateShapeOpsToRunFirstPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, CinnGroupLoweringPass);
