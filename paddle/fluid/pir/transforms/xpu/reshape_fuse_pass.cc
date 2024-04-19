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

#include "paddle/fluid/pir/transforms/xpu/reshape_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

/*
fuse gn + activation block in to group_norm_silu op
For example:
graph:
                      X
              Scale   |   Bias
                   \  |  /
                  group norm
                   /  |  \
                  /   |   \
            variance  |   mean
                      |
                     silu
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      X
              Scale   |   Bias
                   \  |  /
                group_norm_silu
                      |
                     Out
*/

namespace {

class Conv2dBiasPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "Conv2dBiasPattern"; }

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
    add({&pat.Tensor("conv2d_out"), &pat.Tensor("bias")},
        {&pat.Tensor("add_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
        return false;
      }

      std::vector<int64_t> add_bias_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      if (add_bias_shape.size() != 4 && add_bias_shape.at(0) != 1 &&
          add_bias_shape.at(2) != 1 && add_bias_shape.at(3) != 1) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &add_bias_outshape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          std::vector<int64_t> add_bias_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bias"));
          return {add_bias_shape[1]};
        });
    const auto &add_bias_inshape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          std::vector<int64_t> add_bias_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bias"));
          return add_bias_shape;
        });

    const auto &reshape_op = res.Op(paddle::dialect::ReshapeOp::name(),
                                    {{"shape", add_bias_outshape}});
    res.Tensor("out_bias") = reshape_op(res.Tensor("bias"));

    const auto &add_xpu = res.Op(paddle::dialect::AddOp::name());
    add_xpu({&res.Tensor("conv2d_out"), &res.Tensor("out_bias")},
            {&res.Tensor("add_out")});
  }
};

class ReshapeXPUFusePass : public pir::PatternRewritePass {
 public:
  ReshapeXPUFusePass() : pir::PatternRewritePass("reshape_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<Conv2dBiasPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateReshapeXPUFusePass() {
  return std::make_unique<ReshapeXPUFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(reshape_xpu_fuse_pass, ReshapeXPUFusePass);
