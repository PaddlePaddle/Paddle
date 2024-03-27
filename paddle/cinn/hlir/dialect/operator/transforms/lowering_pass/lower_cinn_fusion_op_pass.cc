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
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/broadcast_with_cf.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/pre_analysis.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;
using cinn::dialect::ir::details::FusionOpAnalysis;
using cinn::dialect::ir::details::GetBlockOutsideInput;
using cinn::dialect::ir::details::GetJitKernelAttr;
using cinn::dialect::ir::details::PreAnalysisInfo;

pir::Operation* ProcessDyShapeGroup(
    const OpLoweringGroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
    const PreAnalysisInfo& pre_analysis_info,
    pir::PatternRewriter& rewriter) {  // NOLINT
  // 1. construct broadcast tree
  const auto& broadcast_tree_info =
      pre_analysis_info.broadcast_tree_infos.at(group);
  auto group_inputs = GetBlockOutsideInput(group->ops());
  // has multiple branch
  if (broadcast_tree_info->HasMultiBranch()) {
    std::vector<pir::Type> output_types;
    auto group_output_values = group->GetGroupOutputValues();
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    return CompileBroadcastTreeToConditionBlock(*broadcast_tree_info,
                                                group,
                                                shape_analysis,
                                                group_inputs,
                                                output_types,
                                                rewriter);
  } else {  // no condition block
    // compile group to jit_kernel_op
    std::vector<pir::Type> output_types;
    const auto& group_output_values = group->output_values();
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      auto base_type =
          group_output_values[i].type().dyn_cast<::pir::DenseTensorType>();
      auto dim_info = base_type.dims();
      if (shape_analysis.HasShapeOrDataForValue(group_output_values[i])) {
        auto shape = group->GetShapeOrDataExprs(group_output_values[i]).shape();
        for (size_t k = 0; k < shape.size(); ++k) {
          if (shape[k].isa<int64_t>()) {
            dim_info[k] = shape[k].Get<int64_t>();
          }
        }
      }
      auto new_type = ::pir::DenseTensorType::get(pir::IrContext::Instance(),
                                                  base_type.dtype(),
                                                  dim_info,
                                                  base_type.data_layout(),
                                                  base_type.lod(),
                                                  base_type.offset());
      output_types.push_back(new_type);
    }
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, GetJitKernelAttr(group), output_types);
    return jit_kernel_op;
  }
}
class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  FusionOpPattern(::pir::IrContext* context,
                  const PreAnalysisInfo& pre_analysis_info)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context),
        pre_analysis_info_(pre_analysis_info) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto* program = fusion_op->GetParentProgram();
    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(program);
    VLOG(4) << "Program before lowering: \n"
            << pir::CustomPrintHelper(*program, shape_analysis.PrintHook());

    // TODO(zhangyuqin1998): Replace pir::Group with a new structure
    OpLoweringGroupPtr group = GetGroup(fusion_op);
    pir::Operation* compiled_op = ProcessGroup(group, shape_analysis, rewriter);

    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(fusion_op.result(i), compiled_op->result(i));
      if (shape_analysis.HasShapeOrDataForValue(fusion_op.result(i))) {
        shape_analysis.SetShapeOrDataForValue(
            compiled_op->result(i),
            shape_analysis.GetShapeOrDataForValue(fusion_op.result(i)));
      } else {
        LOG(WARNING) << "No shape_data for "
                     << fusion_op.result(i).defining_op()->name() << "_result_"
                     << i;
      }
    }
    rewriter.EraseOp(fusion_op);
    return true;
  }

 protected:
  virtual const PreAnalysisInfo& GetPreAnalysisInfo() const {
    return pre_analysis_info_;
  }

  virtual OpLoweringGroupPtr GetGroup(cinn::dialect::FusionOp fusion_op) const {
    return pre_analysis_info_.group_infos.at(fusion_op.operation());
  }

  virtual pir::Operation* ProcessGroup(
      const OpLoweringGroupPtr& group,
      pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
      pir::PatternRewriter& rewriter) const {          // NOLINT
    auto group_inputs = GetBlockOutsideInput(group->ops());
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
  const PreAnalysisInfo& pre_analysis_info_;  // not owned
};

class LowerCinnFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context, pre_analysis_info_);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      VLOG(5) << "start to pre-analysis all fusion ops in ModuleOp with static "
                 "shape mode.";
      FusionOpAnalysis(&pre_analysis_info_, /*is_dy_shape=*/false).Run(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable PreAnalysisInfo pre_analysis_info_;
};

class DyShapeFusionOpPattern : public FusionOpPattern {
 public:
  using FusionOpPattern::FusionOpPattern;

 protected:
  virtual pir::Operation* ProcessGroup(
      const OpLoweringGroupPtr& group,
      pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
      pir::PatternRewriter& rewriter) const {          // NOLINT
    return ProcessDyShapeGroup(
        group, shape_analysis, GetPreAnalysisInfo(), rewriter);
  }
};

class LowerCinnDyShapeFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnDyShapeFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_dynamic_shape_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<DyShapeFusionOpPattern>(context, pre_analysis_info_);
    ps.Add<RefreshCombineOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      VLOG(5) << "start to pre-analysis all fusion ops in ModuleOp with "
                 "dynamic shape mode.";
      FusionOpAnalysis(&pre_analysis_info_, /*is_dy_shape=*/true).Run(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable PreAnalysisInfo pre_analysis_info_;
};

}  // namespace

namespace cinn::dialect::ir {
std::unique_ptr<::pir::Pass> CreateLowerCinnFusionOpPass() {
  return std::make_unique<LowerCinnFusionOpPass>();
}

std::unique_ptr<::pir::Pass> CreateLowerCinnDyShapeFusionOpPass() {
  return std::make_unique<LowerCinnDyShapeFusionOpPass>();
}

}  // namespace cinn::dialect::ir

// REGISTER_IR_PASS(cinn_group_lowering, LowerCinnFusionOpPass);
