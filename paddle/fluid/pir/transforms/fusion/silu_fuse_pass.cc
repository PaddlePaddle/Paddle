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

#include "paddle/fluid/pir/transforms/fusion/silu_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class SiluFusePassPattern
    : public paddle::drr::DrrPatternBase<SiluFusePassPattern> {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &sigmoid_op = pat.Op(paddle::dialect::SigmoidOp::name());
    const auto &multiply_op = pat.Op(paddle::dialect::MultiplyOp::name());
    pat.Tensor("sigmoid_out") = sigmoid_op(pat.Tensor("sigmoid_in"));
    pat.Tensor("multiply_out") =
        multiply_op(pat.Tensor("multiply_in"), pat.Tensor("sigmoid_out"));
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &swish_op = res.Op(paddle::dialect::MultiheadMatmulOp::name());
    res.Tensor("multiply_out") = swish_op(pat.Tensor("sigmoid_in"));
  }
};

class SiluFusePass : public pir::PatternRewritePass {
 public:
  SiluFusePass() : pir::PatternRewritePass("silu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(SiluFusePassPattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateSiluFusePass() {
  return std::make_unique<SiluFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(silu_fuse_pass, SiluFusePass);
