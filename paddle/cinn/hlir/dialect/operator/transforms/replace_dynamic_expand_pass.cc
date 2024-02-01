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

#include "paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class DynamicExpandOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ExpandOp> {
 public:
  explicit DynamicExpandOpPattern(pir::IrContext* context)
      : pir::OpRewritePattern<paddle::dialect::ExpandOp>::OpRewritePattern(
            context) {}

  bool MatchAndRewrite(paddle::dialect::ExpandOp op,
                       pir::PatternRewriter& rewriter) const override {
    if (!op->operand_source(1)
             .defining_op()
             ->isa<cinn::dialect::GenerateShapeOp>()) {
      return false;
    }

    const ::pir::Operation* broadcast = [&] {
      int x_rank = op->operand_source(0)
                       .type()
                       .dyn_cast<pir::DenseTensorType>()
                       .dims()
                       .size();
      int out_rank =
          op->result(0).type().dyn_cast<pir::DenseTensorType>().dims().size();
      std::vector<int64_t> broadcast_axes(x_rank, 0);
      size_t index_gap = out_rank - x_rank;
      for (size_t i = 0; i < x_rank; ++i) {
        broadcast_axes[i] = i + index_gap;
      }
      std::vector<int64_t> out_shape(out_rank, -1);
      return rewriter.Build<cinn::dialect::BroadcastOp>(
          op->operand_source(0), broadcast_axes, out_shape);
    }();

    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
    CHECK(shape_analysis.HasShapeOrDataForValue(op.result(0)))
        << "Can't find DimExpr for output of reshape in shape_analysis.";
    shape_analysis.SetShapeOrDataForValue(
        broadcast->result(0),
        shape_analysis.GetShapeOrDataForValue(op.result(0)));

    rewriter.ReplaceAllUsesWith(op->result(0), broadcast->result(0));
    rewriter.EraseOp(op);

    return true;
  }
};

class ReplaceDynamicExpandOpPass : public pir::Pass {
 public:
  ReplaceDynamicExpandOpPass()
      : pir::Pass("replace_dynamic_expand_op_pass", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DynamicExpandOpPattern>(context);
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
            const auto& [_, num_rewrites] =
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

std::unique_ptr<pir::Pass> CreateReplaceDynamicExpandOpPass() {
  return std::make_unique<ReplaceDynamicExpandOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
