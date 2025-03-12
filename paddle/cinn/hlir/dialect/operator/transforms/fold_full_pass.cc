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

#include "paddle/cinn/hlir/dialect/operator/transforms/fold_full_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

template <class OPTYPE>
class FoldFullWithReshapeOpPattern : public pir::OpRewritePattern<OPTYPE> {
 public:
  using pir::OpRewritePattern<OPTYPE>::OpRewritePattern;

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    auto* pre_op = op->operand_source(0).defining_op();
    if (!pre_op || !pre_op->template isa<paddle::dialect::FullOp>()) {
      return false;
    }
    const auto& out_shape =
        op->result(0)
            .type()
            .template dyn_cast<paddle::dialect::DenseTensorType>()
            .dims();
    if (common::contain_unknown_dim(out_shape)) {
      return false;
    }

    pir::AttributeMap attrs = pre_op->attributes();
    attrs["shape"] = paddle::dialect::IntArrayAttribute::get(
        pir::IrContext::Instance(),
        phi::IntArray(out_shape.Get(), out_shape.size()));

    auto new_full_op = rewriter.Build<paddle::dialect::FullOp>(attrs);
    new_full_op->result(0).set_type(op->result(0).type());
    rewriter.ReplaceAllUsesWith(op->result(0), new_full_op->result(0));
    rewriter.EraseOp(op);
    if (pre_op->use_empty()) {
      rewriter.EraseOp(pre_op);
    }

    return true;
  }
};

class FoldFullOpPass : public pir::PatternRewritePass {
 public:
  FoldFullOpPass() : pir::PatternRewritePass("fold_full_ops_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    ps.Add<FoldFullWithReshapeOpPattern<paddle::dialect::ReshapeOp>>(context);
    ps.Add<FoldFullWithReshapeOpPattern<paddle::dialect::TransposeOp>>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateFoldFullOpPass() {
  return std::make_unique<FoldFullOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
