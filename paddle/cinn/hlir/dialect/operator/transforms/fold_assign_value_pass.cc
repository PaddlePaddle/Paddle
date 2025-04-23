// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/fold_assign_value_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class FoldFullAssignValueOpsPattern
    : public pir::OpRewritePattern<paddle::dialect::AssignValue_Op> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::AssignValue_Op>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AssignValue_Op op,
                       pir::PatternRewriter& rewriter) const override {
    auto* pre_op = op.operand_source(0).defining_op();
    if (!pre_op || !pre_op->isa<paddle::dialect::FullOp>() ||
        pre_op->result(0).use_count() > 1) {
      return false;
    }
    auto new_assign_value_op =
        rewriter.Build<paddle::dialect::AssignValueOp>(op.attributes());
    rewriter.ReplaceAllUsesWith(op.result(0), new_assign_value_op->result(0));
    rewriter.EraseOp(op);
    rewriter.EraseOp(pre_op);
    return true;
  }
};

template <typename AssignValueOpType>
class FoldAssignValueCastOpsPattern
    : public pir::OpRewritePattern<AssignValueOpType> {
 public:
  using pir::OpRewritePattern<AssignValueOpType>::OpRewritePattern;

  bool MatchAndRewrite(AssignValueOpType op,
                       pir::PatternRewriter& rewriter) const override {
    if (op.result(0).use_count() != 1) return false;
    pir::Operation* next_op = op.result(0).first_use().owner();
    if (!(next_op->isa<paddle::dialect::CastOp>())) return false;

    auto cast_op = next_op->dyn_cast<paddle::dialect::CastOp>();
    pir::AttributeMap attributes = op.attributes();
    attributes["dtype"] = cast_op.attribute("dtype");

    bool is_inplace_op = op.num_operands() == 1;
    pir::Operation* new_assign_value_op;
    if (is_inplace_op) {
      if (op.operand_source(0).use_count() > 1) return false;
      new_assign_value_op = rewriter.Build<paddle::dialect::AssignValue_Op>(
          op->operand_source(0), attributes);
    } else {
      new_assign_value_op =
          rewriter.Build<paddle::dialect::AssignValueOp>(attributes);
    }
    rewriter.ReplaceAllUsesWith(cast_op->result(0),
                                new_assign_value_op->result(0));
    rewriter.EraseOp(cast_op);
    rewriter.EraseOp(op);
    return true;
  }
};

class FoldAssignValueOpPass : public pir::PatternRewritePass {
 public:
  FoldAssignValueOpPass()
      : pir::PatternRewritePass("fold_assign_value_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<FoldFullAssignValueOpsPattern>(context);
    ps.Add<FoldAssignValueCastOpsPattern<paddle::dialect::AssignValue_Op>>(
        context);
    ps.Add<FoldAssignValueCastOpsPattern<paddle::dialect::AssignValueOp>>(
        context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateFoldAssignValueOpPass() {
  return std::make_unique<FoldAssignValueOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
