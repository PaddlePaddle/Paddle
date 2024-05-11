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
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

bool ReplaceOpWithReshapeOp(pir::Operation* op,
                            pir::ShapeConstraintIRAnalysis* shape_analysis,
                            pir::PatternRewriter& rewriter) {  // NOLINT
  pir::Value output = op->result(0);
  // Try to Get more detail output info
  const auto& GetOutputShape = [&]() -> std::vector<int> {
    std::vector<int> shape = phi::vectorize<int>(
        output.type().dyn_cast<pir::DenseTensorType>().dims());

    const auto& shape_info =
        shape_analysis->GetShapeOrDataForValue(op->result(0)).shape();
    int temp_dim = -1;

    for (size_t i = 0; i < shape_info.size(); ++i) {
      if (shape_info[i].isa<int64_t>()) {
        shape[i] = shape_info[i].Get<int64_t>();
      } else {
        shape[i] = temp_dim;
        temp_dim = 1;
      }
    }
    return shape;
  };

  auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
      op->operand_source(0), GetOutputShape());

  shape_analysis->SetShapeOrDataForValue(
      cinn_reshape.result(0), shape_analysis->GetShapeOrDataForValue(output));

  rewriter.ReplaceAllUsesWith(output, cinn_reshape.result(0));
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
