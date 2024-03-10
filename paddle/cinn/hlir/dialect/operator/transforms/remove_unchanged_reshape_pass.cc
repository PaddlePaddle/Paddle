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

#include "paddle/cinn/hlir/dialect/operator/transforms/remove_unchanged_reshape_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {
using paddle::dialect::details::GetExprVecFromShape;

bool RemoveOp(pir::Operation* op, pir::PatternRewriter* rewriter) {
  const auto& IsDynamicShape = [](const pir::Value& value) -> bool {
    return value.type().dyn_cast<pir::ShapedTypeInterface>().IsDynamicShape();
  };
  const auto& GetDims = [](const pir::Value& value) {
    return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
  };

  pir::Value input = op->operand_source(0);
  pir::Value output = op->result(0);
  const auto& IsSameShape = [&]() -> bool {
    const bool has_dynamic_shape =
        IsDynamicShape(input) || IsDynamicShape(output);
    if (has_dynamic_shape) {
      auto& shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
      if (shape_analysis.HasShapeOrDataForValue(input) &&
          shape_analysis.HasShapeOrDataForValue(output)) {
        auto input_sym_shape =
            GetExprVecFromShape(shape_analysis.GetShapeOrDataForValue(input));
        auto output_sym_shape =
            GetExprVecFromShape(shape_analysis.GetShapeOrDataForValue(output));
        return input_sym_shape == output_sym_shape;
      }
      return false;
    }
    return GetDims(input) == GetDims(output);
  };

  if (IsSameShape()) {
    rewriter->ReplaceAllUsesWith(output, input);
    rewriter->EraseOp(op);
    return true;
  }

  return false;
}

template <typename OPTYPE>
class RemoveUnchangedReshapePattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return RemoveOp(op, &rewriter);
  }
};

class MergeReshapePattern
    : public pir::OpRewritePattern<cinn::dialect::ReshapeOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::ReshapeOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::ReshapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    if (auto pre_shape = op->operand_source(0)
                             .defining_op()
                             ->dyn_cast<cinn::dialect::ReshapeOp>()) {
      op->operand(0).set_source(pre_shape->operand_source(0));

      return true;
    }

    return false;
  }
};

class RemoveUnchangedReshapePass : public pir::PatternRewritePass {
 public:
  RemoveUnchangedReshapePass()
      : pir::PatternRewritePass("remove_unchanged_reshape_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    // remove out_shape equal in_shape reshape op
    ps.Add<RemoveUnchangedReshapePattern<cinn::dialect::ReshapeOp>>(context);
    ps.Add<RemoveUnchangedReshapePattern<paddle::dialect::ReshapeOp>>(context);
    ps.Add<MergeReshapePattern>(context);
    ps.Add<RefreshCombineOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateRemoveUnchangedReshapePass() {
  return std::make_unique<RemoveUnchangedReshapePass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

REGISTER_IR_PASS(remove_unchanged_reshape_pass,
                 ::cinn::dialect::ir::RemoveUnchangedReshapePass);
