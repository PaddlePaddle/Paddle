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
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class RemoveUnchangedReshapePattern
    : public pir::OpRewritePattern<cinn::dialect::ReshapeOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::ReshapeOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::ReshapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto in_dim = op->operand_source(0)
                      .type()
                      .dyn_cast<paddle::dialect::DenseTensorType>()
                      .dims();
    auto out_dim = op->result(0)
                       .type()
                       .dyn_cast<paddle::dialect::DenseTensorType>()
                       .dims();

    auto reshape_op = op->operand_source(0)
                          .dyn_cast<pir::OpResult>()
                          .owner()
                          ->dyn_cast<cinn::dialect::ReshapeOp>();

    if (in_dim == out_dim) {
      rewriter.ReplaceAllUsesWith(op->result(0), op->operand_source(0));
      rewriter.EraseOp(op);
      return true;
    }

    return false;
  }
};

RemoveUnchangedReshapePass::RemoveUnchangedReshapePass()
    : pir::PatternRewritePass("remove_unchanged_reshape_pass", 1) {}

pir::RewritePatternSet RemoveUnchangedReshapePass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);

  // remove out_shape equal in_shape reshape op
  ps.Add<RemoveUnchangedReshapePattern>(context);

  return ps;
}

bool RemoveUnchangedReshapePass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
