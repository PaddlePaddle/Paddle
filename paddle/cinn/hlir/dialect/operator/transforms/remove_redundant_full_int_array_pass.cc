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

#include "paddle/cinn/hlir/dialect/operator/transforms/remove_redundant_full_int_array_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class RemoveRedundantFullIntArrayPattern
    : public pir::OpRewritePattern<paddle::dialect::FullIntArrayOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FullIntArrayOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::FullIntArrayOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto block = op->GetParent();
    if (!block) return false;
    pir::AttributeMap attrs = op->attributes();
    auto dtype = attrs.at("dtype");
    auto value = attrs.at("value");
    auto place = attrs.at("place");

    for (auto other_op = block->begin(); other_op != block->end(); ++other_op) {
      if (!other_op->isa<paddle::dialect::FullIntArrayOp>()) continue;
      if (op.operation() == &(*other_op)) break;
      pir::AttributeMap other_attrs = other_op->attributes();
      if (dtype != other_attrs.at("dtype") || place != other_attrs.at("place"))
        continue;
      auto other_value = other_attrs.at("value");
      if (!other_value) continue;
      if (value == other_value) {
        rewriter.ReplaceAllUsesWith(op->result(0), other_op->result(0));
        rewriter.EraseOp(op);
        return true;
      }
    }
    return false;
  }
};

class RemoveRedundantFullIntArrayPass : public pir::PatternRewritePass {
 public:
  RemoveRedundantFullIntArrayPass()
      : pir::PatternRewritePass("remove_redundant_full_int_array_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<RemoveRedundantFullIntArrayPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateRemoveRedundantFullIntArrayPass() {
  return std::make_unique<RemoveRedundantFullIntArrayPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
