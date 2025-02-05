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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/lower_cinn_fusion_op_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/pre_analysis.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace cinn::dialect::ir::details {
class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  FusionOpPattern(::pir::IrContext* context, const GroupInfoMap& group_infos)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context),
        group_infos_(group_infos) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto* program = fusion_op->GetParentProgram();
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(program);

    // TODO(zhangyuqin1998): Replace pir::Group with a new structure
    OpLoweringGroupPtr group = GetGroup(fusion_op);
    pir::Operation* compiled_op = ProcessGroup(group, rewriter);

    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(fusion_op.result(i), compiled_op->result(i));
      shape_analysis.SetShapeOrDataForValue(
          compiled_op->result(i),
          shape_analysis.GetShapeOrDataForValue(fusion_op.result(i)));
    }
    rewriter.EraseOp(fusion_op);
    return true;
  }

 protected:
  virtual OpLoweringGroupPtr GetGroup(cinn::dialect::FusionOp fusion_op) const {
    return group_infos_.at(fusion_op.operation());
  }

  virtual pir::Operation* ProcessGroup(
      const OpLoweringGroupPtr& group,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    auto group_inputs =
        cinn::hlir::framework::pir::GetBlockOutsideInput(group->ops());
    // compile group to jit_kernel_op
    std::vector<pir::Type> output_types;
    const auto& group_output_values = group->output_values();
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, GetJitKernelAttr(group), output_types);
    return jit_kernel_op;
  }

 private:
  const GroupInfoMap& group_infos_;  // not owned
};

class LowerCinnFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context, group_infos_);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      VLOG(5) << "start to pre-analysis all fusion ops in ModuleOp with static "
                 "shape mode.";
      FusionOpAnalysis(&group_infos_, /*is_dy_shape=*/false).Run(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable GroupInfoMap group_infos_;
};

class LowerCinnDyShapeFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnDyShapeFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_dynamic_shape_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context, group_infos_);
    ps.Add<RefreshCombineOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      VLOG(5) << "start to pre-analysis all fusion ops in ModuleOp with "
                 "dynamic shape mode.";
      FusionOpAnalysis(&group_infos_, /*is_dy_shape=*/true).Run(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable GroupInfoMap group_infos_;
};

}  // namespace cinn::dialect::ir::details

namespace cinn::dialect::ir {
std::unique_ptr<::pir::Pass> CreateLowerCinnFusionOpPass() {
  return std::make_unique<details::LowerCinnFusionOpPass>();
}

std::unique_ptr<::pir::Pass> CreateLowerCinnDyShapeFusionOpPass() {
  return std::make_unique<details::LowerCinnDyShapeFusionOpPass>();
}

}  // namespace cinn::dialect::ir

// REGISTER_IR_PASS(cinn_group_lowering, LowerCinnFusionOpPass);
