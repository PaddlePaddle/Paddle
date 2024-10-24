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
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

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

      pir::ShapeConstraintIRAnalysis& shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

      const auto& GetOutputShapeByDimExpr = [&]() -> std::vector<int64_t> {
        std::vector<int64_t> out_shape(out_rank, -1);
        auto shape_info =
            shape_analysis.GetShapeOrDataForValue(op->result(0)).shape();

        for (size_t i = 0; i < shape_info.size(); ++i) {
          if (shape_info[i].isa<int64_t>()) {
            out_shape[i] = shape_info[i].Get<int64_t>();
          }
        }
        return out_shape;
      };

      auto out_shape = GetOutputShapeByDimExpr();

      return rewriter.Build<cinn::dialect::BroadcastOp>(
          op->operand_source(0), broadcast_axes, out_shape);
    }();

    if (auto pre_full = broadcast->operand_source(0)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullOp>()) {
      auto input_dim = pre_full.result(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims();
    }

    rewriter.ReplaceAllUsesWith(op->result(0), broadcast->result(0));
    rewriter.EraseOp(op);

    return true;
  }
};

class ReplaceDynamicExpandOpPass : public pir::PatternRewritePass {
 public:
  ReplaceDynamicExpandOpPass()
      : pir::PatternRewritePass("replace_dynamic_expand_op_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DynamicExpandOpPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<cinn::dialect::GroupOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateReplaceDynamicExpandOpPass() {
  return std::make_unique<ReplaceDynamicExpandOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
