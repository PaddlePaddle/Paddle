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

#include "paddle/fluid/pir/transforms/gpu/silu_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class SiluFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SiluFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &sigmoid_op = pat.Op(paddle::dialect::SigmoidOp::name());
    const auto &multiply_op = pat.Op(paddle::dialect::MultiplyOp::name());
    pat.Tensor("sigmoid_out") = sigmoid_op(pat.Tensor("sigmoid_in"));
    pat.Tensor("multiply_out") =
        multiply_op(pat.Tensor("sigmoid_in"), pat.Tensor("sigmoid_out"));
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &swish_op = res.Op(paddle::dialect::SwishOp::name());
    res.Tensor("multiply_out") = swish_op(res.Tensor("sigmoid_in"));
  }
};

class SiluFusePass : public pir::PatternRewritePass {
 public:
  SiluFusePass() : pir::PatternRewritePass("silu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<SiluFusePattern>(context));
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
