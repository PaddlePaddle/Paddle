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
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class DynamicReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  explicit DynamicReshapeOpPattern(pir::IrContext* context)
      : pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern(
            context) {}

  bool MatchAndRewrite(paddle::dialect::ReshapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto scale_factor_gen_op =
        op->operand_source(1).dyn_cast<pir::OpResult>().owner();
    auto output = op.result(0);

    // The value of shape attribute is fake, we only use the output shape info
    // in shape analysis.
    std::vector<int> shape(
        output.type().dyn_cast<pir::ShapedTypeInterface>().GetRank(), 1);
    shape[0] = -1;

    auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
        op->operand_source(0).dyn_cast<pir::OpResult>(), shape);

    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    CHECK(shape_analysis.HasShapeOrDataForValue(output))
        << "Can't find DimExpr for output of reshape in shape_analysis.";
    const auto& out_origin_expr_shape =
        shape_analysis.GetShapeOrDataForValue(output);
    shape_analysis.SetShapeOrDataForValue(cinn_reshape.result(0),
                                          out_origin_expr_shape);

    for (auto out : op->results()) {
      VLOG(0) << " OUT:";
      for (auto it = out.use_begin(); it != out.use_end(); ++it) {
        VLOG(0) << " user: " << it->owner()->name();
      }
    }
    rewriter.ReplaceAllUsesWith(output, cinn_reshape.result(0));
    rewriter.EraseOp(op);

    return true;
  }
};

class DynamicReshapeOpPass : public pir::Pass {
 public:
  DynamicReshapeOpPass()
      : pir::Pass("cinn_dynamic_reshape_op_pass", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DynamicReshapeOpPattern>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        for (auto& op : block) {
          if (op.isa<cinn::dialect::FusionOp>()) {
            auto [_, num_rewrites] =
                pir::ApplyPatternsGreedily(&op, patterns_, cfg);
            AddStatistics(num_rewrites);
          }
        }
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

std::unique_ptr<pir::Pass> CreateDynamicReshapeOpPass() {
  return std::make_unique<DynamicReshapeOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
