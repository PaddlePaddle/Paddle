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

#include "paddle/fluid/pir/transforms/onednn/squeeze_transpose_onednn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class SqueezeTransposePattern : public paddle::drr::DrrPatternBase {
 public:
  SqueezeTransposePattern() = default;

  std::string name() const override { return "SqueezeTransposePattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &squeeze = pat.Op(paddle::dialect::SqueezeOp::name());
    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});

    squeeze({&pat.Tensor("x"), &full_1()}, {&pat.Tensor("squeeze_out")});

    const auto &transpose = pat.Op(paddle::dialect::TransposeOp::name(),
                                   {{"perm", pat.Attr("perm")}});

    transpose({&pat.Tensor("squeeze_out")}, {&pat.Tensor("transpose_op_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto axis = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
      auto perm = match_ctx.Attr<std::vector<int>>("perm");
      if (perm.size() <= 0) return false;
      if (axis.size() <= 0) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_reshape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    const auto &fused_transpose =
        res.Op(paddle::onednn::dialect::FusedTransposeOp::name(),
               {{
                   {"axis", pat.Attr("perm")},
                   {"fused_squeeze2_axes", fused_reshape_attr},
                   {"fused_unsqueeze2_axes", res.VectorInt32Attr({})},
                   {"fused_reshape2_shape", res.VectorInt32Attr({})},
                   {"scale", res.Float32Attr(1.0f)},
                   {"shift", res.Float32Attr(0.0f)},
                   {"output_data_type", res.StrAttr("fp32")},
                   {"data_format", res.StrAttr("AnyLayout")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
               }});
    fused_transpose({&res.Tensor("x")}, {&res.Tensor("transpose_op_out")});
  }
};

class SqueezeTransposePass : public pir::PatternRewritePass {
 public:
  SqueezeTransposePass()
      : pir::PatternRewritePass("squeeze_transpose_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<SqueezeTransposePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateSqueezeTransposeOneDNNPass() {
  // pd_op.squeeze + transpose2  -> onednn_op.fused_transpose
  return std::make_unique<SqueezeTransposePass>();
}

}  // namespace pir

REGISTER_IR_PASS(squeeze_transpose_onednn_fuse_pass, SqueezeTransposePass);
