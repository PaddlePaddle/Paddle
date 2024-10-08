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

#include "paddle/fluid/pir/transforms/general/add_shadow_output_after_dead_parameter_pass.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class AddShadowOutputAfterDeadParameterPattern
    : public pir::OpRewritePattern<pir::ParameterOp> {
 public:
  using pir::OpRewritePattern<pir::ParameterOp>::OpRewritePattern;
  bool MatchAndRewrite(
      pir::ParameterOp op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    if (!op->use_empty()) {
      return false;
    }
    rewriter.SetInsertionPointToBlockEnd(op->GetParent());
    rewriter.Build<pir::ShadowOutputOp>(op->result(0), op.param_name());
    return true;
  }
};

class AddShadowOutputAfterDeadParameterPass : public pir::PatternRewritePass {
 public:
  AddShadowOutputAfterDeadParameterPass()
      : pir::PatternRewritePass("add_shadow_output_after_dead_parameter_pass",
                                0) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddShadowOutputAfterDeadParameterPattern>(context);
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

std::unique_ptr<pir::Pass> CreateAddShadowOutputAfterDeadParameterPass() {
  return std::make_unique<AddShadowOutputAfterDeadParameterPass>();
}

}  // namespace pir

REGISTER_IR_PASS(add_shadow_output_after_dead_parameter_pass,
                 AddShadowOutputAfterDeadParameterPass);
