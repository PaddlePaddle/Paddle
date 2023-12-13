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

#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_add_fuse_pass.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"

namespace {

class Conv2dAddFusePattern
    : public pir::drr::DrrPatternBase<Conv2dAddFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("bias"));

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add_act = res.Op(
        paddle::dialect::FusedConv2dAddActOp::name(),
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"activation",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::string { return "identity"; })},
            {"split_channels",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::vector<int> { return {}; })},
            {"exhaustive_search",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
               return false;
             })},
            {"workspace_size_MB",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> int {
               return 32;
             })},
            {"fuse_alpha",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
               return 0.0f;
             })},
        }});

    fused_conv2d_add_act({&res.Tensor("input"),
                          &res.Tensor("filter"),
                          &res.Tensor("bias"),
                          &res.NoneTensor()},
                         {&res.Tensor("add_out")});
  }
};

class Conv2dAddFusePass : public pir::PatternRewritePass {
 public:
  Conv2dAddFusePass() : pir::PatternRewritePass("conv2d_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(Conv2dAddFusePattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dAddFusePass() {
  return std::make_unique<Conv2dAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(conv2d_add_fuse_pass, Conv2dAddFusePass);
