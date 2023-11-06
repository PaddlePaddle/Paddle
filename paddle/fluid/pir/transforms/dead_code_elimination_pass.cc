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

#include "paddle/fluid/pir/transforms/dead_code_elimination_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class DeadCodeElimination : public pir::RewritePattern {
 public:
  DeadCodeElimination(pir::IrContext* context,
                      pir::PatternBenefit benefit = 1,
                      const std::vector<std::string>& generated_names = {})
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context, generated_names) {
  }

  bool Match(pir::Operation* op) const override {
    if (op->isa<paddle::dialect::FetchOp>() ||
        op->isa<paddle::dialect::ShadowOutputOp>())
      return false;

    bool use_empty = true;
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      use_empty &= op->result(i).use_empty();
    }
    return use_empty;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    if (op->dyn_cast<pir::GetParameterOp>()) {
      // Delete parameter from program.
      pir::GetParameterOp get_parameter_op =
          op->dyn_cast<pir::GetParameterOp>();
      get_parameter_op->GetParentProgram()->parameters().erase(
          get_parameter_op->attributes()
              .at(get_parameter_op.attributes_name[0])
              .dyn_cast<pir::StrAttribute>()
              .AsString());
    }
    rewriter.eraseOp(op);
  }
};

class DeadCodeEliminationPass : public pir::Pass {
 public:
  DeadCodeEliminationPass() : pir::Pass("dead_code_elimination_pass", 0) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DeadCodeElimination>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(dead_code_elimination_pass, DeadCodeEliminationPass);
