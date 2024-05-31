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

#include "paddle/fluid/pir/transforms/general/map_op_to_another_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DepthWiseConv2d2Conv2dPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "DepthWiseConv2d2Conv2dPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &depthwise_conv2d_op =
        pat.Op(paddle::dialect::DepthwiseConv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    depthwise_conv2d_op({&pat.Tensor("input"), &pat.Tensor("filter")},
                        {&pat.Tensor("depthwise_conv2d_out")});
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 8100
      auto groups = match_ctx.Attr<int>("groups");
      return groups > 1;
#else
      return false;
#endif
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &conv2d =
        res.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    conv2d({&res.Tensor("input"), &res.Tensor("filter")},
           {&res.Tensor("depthwise_conv2d_out")});
  }
};

class MapOpToAnotherPass : public pir::PatternRewritePass {
 public:
  MapOpToAnotherPass() : pir::PatternRewritePass("map_op_to_another_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DepthWiseConv2d2Conv2dPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMapOpToAnotherPass() {
  return std::make_unique<MapOpToAnotherPass>();
}
}  // namespace pir

REGISTER_IR_PASS(map_op_to_another_pass, MapOpToAnotherPass);
