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

#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

bool ReplaceOpWithReshapeOp(pir::Operation* op,
                            pir::ShapeConstraintIRAnalysis* shape_analysis,
                            pir::PatternRewriter& rewriter,  // NOLINT
                            bool with_xshape) {
  pir::Value input = op->operand_source(0);
  pir::Value output = op->result(0);
  const auto& input_shape =
      shape_analysis->GetShapeOrDataForValue(input).shape();
  const auto& output_shape =
      shape_analysis->GetShapeOrDataForValue(output).shape();

  std::vector<pir::Attribute> output_dim_expr_attrs{};
  GenerateShapeOp::SymbolBindings symbol_bindings{};

  int64_t local_dim_expr_id = 0;
  for (unsigned output_dim_idx = 0, input_dim_idx = 0;
       output_dim_idx < output_shape.size();
       ++output_dim_idx) {
    const auto& dim_expr = output_shape.at(output_dim_idx);
    if (dim_expr.isa<int64_t>()) {
      output_dim_expr_attrs.emplace_back(
          ConvertDimExprToAttribute(rewriter.ir_context(), dim_expr));
      continue;
    }
    for (int next_input_dim_idx = input_dim_idx;
         next_input_dim_idx < input_shape.size();
         ++next_input_dim_idx) {
      const auto& input_dim_expr = input_shape.at(next_input_dim_idx);
      if (dim_expr == input_dim_expr) {
        std::string sym_name = ToString(dim_expr);
        if (!dim_expr.isa<std::string>()) {
          sym_name = "SS" + std::to_string(local_dim_expr_id++);
        }
        output_dim_expr_attrs.emplace_back(ConvertDimExprToAttribute(
            rewriter.ir_context(), ::symbol::DimExpr(sym_name)));
        symbol_bindings.emplace_back(GenerateShapeOp::ShapeSymbolBinding{
            sym_name, 0, next_input_dim_idx});
        input_dim_idx = next_input_dim_idx + 1;
      }
    }
  }
  auto out_type = paddle::dialect::DenseTensorType::get(
      rewriter.ir_context(),
      pir::Int64Type::get(rewriter.ir_context()),
      ::common::make_ddim(
          {static_cast<int64_t>(output_dim_expr_attrs.size())}));
  auto cinn_generate_shape = rewriter.Build<cinn::dialect::GenerateShapeOp>(
      std::vector<pir::Value>{input},
      output_dim_expr_attrs,
      symbol_bindings,
      out_type);
  auto pd_reshape = rewriter.Build<paddle::dialect::ReshapeOp>(
      op->operand_source(0), cinn_generate_shape.result(0));

  rewriter.ReplaceAllUsesWith(output, pd_reshape.result(0));
  rewriter.EraseOp(op);
  return true;
}

class DynamicReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  explicit DynamicReshapeOpPattern(pir::IrContext* context)
      : pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern(
            context) {}

  bool MatchAndRewrite(paddle::dialect::ReshapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter, true);
  }
};

class DynamicSqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SqueezeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SqueezeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    const auto& axis_shape_expr =
        shape_analysis.GetShapeOrDataForValue(op.axis());
    PADDLE_ENFORCE_EQ(axis_shape_expr.data().has_value(),
                      true,
                      ::common::errors::PreconditionNotMet(
                          "The axis_shape_expr data must have a value."));

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter, true);
  }
};

class DynamicUnsqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::UnsqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::UnsqueezeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::UnsqueezeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    const auto& axis_shape_expr =
        shape_analysis.GetShapeOrDataForValue(op.axis());
    PADDLE_ENFORCE_EQ(axis_shape_expr.data().has_value(),
                      true,
                      ::common::errors::PreconditionNotMet(
                          "The axis_shape_expr data must have a value."));

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter, true);
  }
};

class DynamicReshapeOpPass : public pir::PatternRewritePass {
 public:
  DynamicReshapeOpPass()
      : pir::PatternRewritePass("cinn_dynamic_reshape_op_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    // Comment out the DynamicReshapeOpPattern to use pd_op.reshape in
    // cinn.group ps.Add<DynamicReshapeOpPattern>(context);
    ps.Add<DynamicSqueezeOpPattern>(context);
    ps.Add<DynamicUnsqueezeOpPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<cinn::dialect::GroupOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateDynamicReshapeOpPass() {
  return std::make_unique<DynamicReshapeOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
