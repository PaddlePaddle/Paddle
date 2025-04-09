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

#include "paddle/fluid/pir/transforms/onednn/fc_onednn_enable_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FcOneDNNEnablePattern : public paddle::drr::DrrPatternBase {
 public:
  FcOneDNNEnablePattern() = default;

  std::string name() const override { return "FcOneDNNEnablePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &fc = pat.Op(paddle::dialect::FcOp::name(),
                            {{"in_num_col_dims", pat.Attr("in_num_col_dims")},
                             {"activation_type", pat.Attr("activation_type")},
                             {"padding_weights", pat.Attr("padding_weights")}});

    fc({&pat.Tensor("input"), &pat.Tensor("weight"), &pat.Tensor("bias")},
       {&pat.Tensor("Out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto input_shape = pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto input_dims = input_shape.size();
      bool support_dims = (input_dims >= 2 || input_shape.size() <= 4);
      constexpr size_t height_axis = 2;
      constexpr size_t width_axis = 3;
      bool support_size = input_dims == 4 ? (input_shape[width_axis] == 1 &&
                                             input_shape[height_axis] == 1)
                                          : true;
      if (!support_dims || !support_size) return false;
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto act_type = match_ctx.Attr<std::string>("activation_type");
      if (!(act_type == "" || act_type == "relu")) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"in_num_col_dims", pat.Attr("in_num_col_dims")},
        {"activation_type", pat.Attr("activation_type")},
        {"padding_weights", pat.Attr("padding_weights")},
        {"use_quantizer", res.BoolAttr(false)},
        {"mkldnn_data_type", res.StrAttr("float32")},
        {"scale_in", res.Float32Attr(1.0f)},
        {"scale_weights", res.VectorFloatAttr({1.0f})},
        {"scale_out", res.Float32Attr(1.0f)},
        {"force_fp32_output", res.BoolAttr(false)},
        {"fuse_activation", res.StrAttr("")},
        {"fuse_alpha", res.Float32Attr(0.0f)},
        {"fuse_beta", res.Float32Attr(0.0f)},
        {"fused_output_scale", res.Float32Attr(1.0f)},
        {"fused_reshape2_shape", res.VectorInt32Attr({})}};

    const auto &fused_fc =
        res.Op(paddle::onednn::dialect::FcOp::name(), fused_attrs);

    fused_fc({&res.Tensor("input"), &res.Tensor("weight"), &res.Tensor("bias")},
             {&res.Tensor("Out")});
  }
};

class FcOneDNNEnablePass : public pir::PatternRewritePass {
 public:
  FcOneDNNEnablePass() : pir::PatternRewritePass("fc_onednn_enable_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FcOneDNNEnablePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcOneDNNEnablePass() {
  // pd_op.fc -> onednn_op.fc
  return std::make_unique<FcOneDNNEnablePass>();
}
}  // namespace pir

REGISTER_IR_PASS(fc_onednn_enable_pass, FcOneDNNEnablePass);
