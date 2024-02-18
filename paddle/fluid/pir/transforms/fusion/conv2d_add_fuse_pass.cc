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

#include "paddle/fluid/pir/transforms/fusion/conv2d_add_fuse_pass.h"

#include <string>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Conv2dAddFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "Conv2dAddFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
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
    pat.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          auto padding_algorithm =
              match_ctx.Attr<std::string>("padding_algorithm");
          if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
              padding_algorithm != "VALID") {
            return false;
          }
          auto groups = match_ctx.Attr<int>("groups");
          if (groups < 1) {
            return false;
          }
          auto data_format = match_ctx.Attr<std::string>("data_format");
          if (data_format != "NCHW" && data_format != "AnyLayout") {
            return false;
          }
          return true;
        });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_conv2d_add_act =
        res.Op(paddle::dialect::FusedConv2dAddActOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"activation", res.StrAttr("identity")},
                   {"split_channels", res.VectorInt32Attr({})},
                   {"exhaustive_search", res.BoolAttr(false)},
                   {"workspace_size_MB", res.Int32Attr(32)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
               }});

    fused_conv2d_add_act({&res.Tensor("input"),
                          &res.Tensor("filter"),
                          &res.Tensor("bias"),
                          &res.InputNoneTensor()},
                         {&res.Tensor("add_out")});
  }
};

class Conv2dAddFusePass : public pir::PatternRewritePass {
 public:
  Conv2dAddFusePass() : pir::PatternRewritePass("conv2d_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<Conv2dAddFusePattern>(context));
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
