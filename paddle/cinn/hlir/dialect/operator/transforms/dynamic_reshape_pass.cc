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
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

std::vector<pir::Value> ReplaceGenerateShapeOpToIdentity(
    const cinn::dialect::GenerateShapeOp& op,
    pir::ShapeConstraintIRAnalysis* shape_analysis,
    pir::PatternRewriter& rewriter) {  // NOLINT
  std::vector<pir::Value> outputs;
  rewriter.SetInsertionPointToBlockEnd(op->GetParent());
  for (const auto& arg : op->operands_source()) {
    auto identity = rewriter.Build<cinn::dialect::IdentityOp>(arg);
    shape_analysis->SetShapeOrDataForValue(
        identity.result(0), shape_analysis->GetShapeOrDataForValue(arg));
    outputs.push_back(identity.result(0));
  }
  return outputs;
}

void HoldShapeValue(pir::Operation* op,
                    pir::ShapeConstraintIRAnalysis* shape_analysis,
                    pir::PatternRewriter& rewriter) {  // NOLINT
  pir::Value shape_value = op->operand_source(1);
  if (shape_value.use_count() > 1) {
    return;
  }

  auto* block = op->GetParent();
  auto yield_op = block->back().dyn_cast<::pir::YieldOp>();
  CHECK(yield_op);
  const std::vector<pir::Value> new_outputs = [&] {
    std::vector<pir::Value> inputs;
    for (const auto& arg : yield_op->operands_source()) {
      inputs.push_back(arg);
    }
    auto generate_shape_op =
        shape_value.defining_op()->dyn_cast<cinn::dialect::GenerateShapeOp>();
    CHECK(generate_shape_op);
    auto new_identity_outputs = ReplaceGenerateShapeOpToIdentity(
        generate_shape_op, shape_analysis, rewriter);
    inputs.insert(
        inputs.end(), new_identity_outputs.begin(), new_identity_outputs.end());
    return inputs;
  }();
  VLOG(0) << "###### new_outputs size: " << new_outputs.size();
  rewriter.SetInsertionPointToBlockEnd(block);
  auto new_yield = rewriter.Build<::pir::YieldOp>(new_outputs);
  rewriter.EraseOp(yield_op);
}

bool ReplaceOpWithReshapeOp(pir::Operation* op,
                            pir::ShapeConstraintIRAnalysis* shape_analysis,
                            pir::PatternRewriter& rewriter) {  // NOLINT
  pir::Value output = op->result(0);
  // Try to Get more detail output info
  const auto& GetOupputShape = [&]() -> std::vector<int> {
    std::vector<int> shape = phi::vectorize<int>(
        output.type().dyn_cast<pir::DenseTensorType>().dims());

    if (shape_analysis->HasShapeOrDataForValue(op->result(0))) {
      auto shape_info =
          shape_analysis->GetShapeOrDataForValue(op->result(0)).shape();

      for (size_t i = 0; i < shape_info.size(); ++i) {
        if (shape_info[i].isa<int64_t>()) {
          shape[i] = shape_info[i].Get<int64_t>();
        } else {
          shape[i] = 1;
        }
      }
    }
    return shape;
  };

  auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
      op->operand_source(0), GetOupputShape());

  shape_analysis->SetShapeOrDataForValue(
      cinn_reshape.result(0), shape_analysis->GetShapeOrDataForValue(output));

  rewriter.ReplaceAllUsesWith(output, cinn_reshape.result(0));
  HoldShapeValue(op, shape_analysis, rewriter);
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

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter);
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
    CHECK(axis_shape_expr.data().has_value());

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter);
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
    CHECK(axis_shape_expr.data().has_value());

    return ReplaceOpWithReshapeOp(op, &shape_analysis, rewriter);
  }
};

class DynamicReshapeOpPass : public pir::PatternRewritePass {
 public:
  DynamicReshapeOpPass()
      : pir::PatternRewritePass("cinn_dynamic_reshape_op_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DynamicReshapeOpPattern>(context);
    ps.Add<DynamicSqueezeOpPattern>(context);
    ps.Add<DynamicUnsqueezeOpPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<cinn::dialect::FusionOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateDynamicReshapeOpPass() {
  return std::make_unique<DynamicReshapeOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
