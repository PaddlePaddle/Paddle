/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/extension.h"

namespace {

class ReluReplacePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "ReluReplacePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &relu = pat.Op("pd_op.relu");
    relu({&pat.Tensor("in")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &custom_relu = res.Op("custom_op.custom_relu");
    custom_relu({&res.Tensor("in")}, {&res.Tensor("out")});
  }
};

class ReluReplacePass : public pir::PatternRewritePass {
 public:
  ReluReplacePass() : pir::PatternRewritePass("relu_replace_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ReluReplacePattern>(context));
    return ps;
  }
};

}  // namespace

REGISTER_IR_PASS(relu_replace_pass, ReluReplacePass);
