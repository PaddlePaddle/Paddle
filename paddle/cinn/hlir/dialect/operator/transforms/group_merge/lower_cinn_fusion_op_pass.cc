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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/lower_cinn_fusion_op_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace {

using cinn::dialect::ir::Group;
using cinn::hlir::framework::pir::CompatibleInfo;

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      block_inner_output.insert(op_list[k]->result(i));
    }
  }

  std::unordered_set<::pir::Value> insert_value;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i)) &&
          !insert_value.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
        insert_value.insert(op_list[k]->operand_source(i));
      }
    }
  }
  return vec_res;
}

std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
CreateGroupShapeOrDataExprs(const cinn::dialect::ir::GroupPtr& group,
                            pir::ShapeConstraintIRAnalysis* shape_analysis) {
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs> value2shape;
  for (auto* op : group->ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto operand = op->operand_source(i);
      if (shape_analysis->HasShapeOrDataForValue(operand)) {
        value2shape.insert(
            {operand, shape_analysis->GetShapeOrDataForValue(operand)});
      }
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      if (value2shape.find(result) == value2shape.end() &&
          shape_analysis->HasShapeOrDataForValue(result)) {
        value2shape.insert(
            {result, shape_analysis->GetShapeOrDataForValue(result)});
      }
    }
  }
  return value2shape;
}

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();
    // TODO(Aurelius84): Remove scope after cleaning PirCompiler usless Build
    // Interface
    auto scope = std::make_shared<cinn::hlir::framework::Scope>();
    auto* program = fusion_op->GetParentProgram();
    auto ir_compiler = cinn::hlir::framework::PirCompilerManager::Create(
        *program, target, scope);
    auto group = RebuildGroup(fusion_op);
    // Because the group is rebuilt, the order of group.output_values generated
    // by BuildCUDAJITInfo may not be same with the order bound in the yield op,
    // so a mapping is required.
    std::unordered_map<::pir::Value, size_t> value2id;

    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
        fusion_op->GetParentProgram());
    group->value_to_shape_or_data_exprs =
        CreateGroupShapeOrDataExprs(group, &shape_analysis);
    if (FLAGS_cinn_enable_map_expr) {
      cinn::adt::TryGenerateMapExprFromGroup(group);
    }

    // TODO(zhangyuqin1998): Replace pir::Group with a new structure
    auto fn_ptr_res = ir_compiler->BuildCUDAJITInfo({group});
    std::unordered_map<std::string, ::pir::Attribute> op_attrs{
        {cinn::dialect::JitKernelOp::kAttrName,
         cinn::dialect::CINNKernelInfoAttribute::get(ctx, fn_ptr_res[0])},
    };

    // Generate jit kernel op input and output
    auto vec_ins = GetBlockOutsideInput(group->ops);

    std::vector<pir::Type> vec_types;
    for (size_t i = 0; i < group->output_values.size(); ++i) {
      vec_types.push_back(group->output_values[i].type());
      value2id[group->output_values[i]] = i;
    }

    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        vec_ins, op_attrs, vec_types);

    auto yeild_op = fusion_op.GetOperators().back();
    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(
          fusion_op.result(i),
          jit_kernel_op.result(value2id[yeild_op->operand_source(i)]));
    }

    rewriter.EraseOp(fusion_op);
    return true;
  }

 private:
  std::shared_ptr<Group> RebuildGroup(cinn::dialect::FusionOp fusion_op) const {
    auto group = std::make_shared<Group>();
    group->op_pattern_kind = cinn::hlir::framework::OpPatternKind::kElementWise;

    // Rebuild ops of the group
    for (auto op : fusion_op.GetOperators()) {
      if (!op->isa<::pir::YieldOp>()) {
        group->ops.push_back(op);
        group->ops_set.insert(op);
        group->op_pattern_kind =
            static_cast<int>(CompatibleInfo::OpKind(*op)) >
                    static_cast<int>(group->op_pattern_kind)
                ? CompatibleInfo::OpKind(*op)
                : group->op_pattern_kind;
      }
    }

    // Rebuild output_ops and input_ops of the group
    auto yeild_op = fusion_op.GetOperators().back();
    for (size_t i = 0; i < yeild_op->num_operands(); ++i) {
      group->output_ops.insert(yeild_op->operand_source(i).defining_op());
    }

    // Rebuild other informations
    // TODO(zhangyuqin1998): Do we need group.master_ops?
    return group;
  }
};

class LowerCinnFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateLowerCinnFusionOpPass() {
  return std::make_unique<LowerCinnFusionOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, LowerCinnFusionOpPass);
