// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/general/replace_fetch_with_shadow_output_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class ReplaceFetchWithShadowOutputPattern
    : public pir::OpRewritePattern<paddle::dialect::FetchOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FetchOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::FetchOp op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    // for pd_op.data -> value -> pd_op.fetch, we insert pd_op.scale before
    // pd_op.fetch to solve error likes [what(): (NotFound) Variable 'xxx' is
    // not found in scope.]
    if (pir::GetDefiningOpForInput(op, 0)->HasAttribute("name")) {
      auto scale_op = rewriter.Build<paddle::dialect::ScaleOp>(
          op->operand_source(0), 1.0, 0.0, true);
      rewriter.Build<pir::ShadowOutputOp>(
          scale_op->result(0),
          op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString());
    } else {
      rewriter.Build<pir::ShadowOutputOp>(
          op->operand_source(0),
          op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString());
    }
    rewriter.EraseOp(op);
    return true;
  }
};

class ReplaceFetchWithShadowOutputPass : public pir::PatternRewritePass {
 public:
  ReplaceFetchWithShadowOutputPass()
      : pir::PatternRewritePass("replace_fetch_with_shadow_output_pass", 0) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ReplaceFetchWithShadowOutputPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<pir::Pass> CreateReplaceFetchWithShadowOutputPass() {
  return std::make_unique<ReplaceFetchWithShadowOutputPass>();
}

}  // namespace pir

REGISTER_IR_PASS(replace_fetch_with_shadow_output_pass,
                 ReplaceFetchWithShadowOutputPass);
