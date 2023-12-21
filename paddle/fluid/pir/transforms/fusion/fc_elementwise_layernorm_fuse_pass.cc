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

#include "paddle/fluid/pir/transforms/fusion/fc_elementwise_layernorm_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class FcElementwiseLayerNormFusePattern
    : public pir::drr::DrrPatternBase<FcElementwiseLayerNormFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"use_mkldnn", pat.Attr("use_mkldnn")},
                   {"padding_weights", pat.Attr("padding_weights")},
                   {"use_quantizer", pat.Attr("use_quantizer")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"scale_in", pat.Attr("scale_in")},
                   {"scale_weights", pat.Attr("scale_weights")},
                   {"scale_out", pat.Attr("scale_out")},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
               });
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias0")},
       {&pat.Tensor("fc_out")});
    add({&pat.Tensor("fc_out"), &pat.Tensor("y")}, {&pat.Tensor("add_out")});
    layernorm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias1")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});
    // Constrains the activation is none
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      int64_t layer_norm_x = 1;
      for (int i = match_ctx.Attr<int>("begin_norm_axis");
           i < match_ctx.Tensor("fc_out").Shape().size();
           i++) {
        layer_norm_x *= match_ctx.Tensor("fc_out").Shape().at(i);
      }
      if (layer_norm_x == match_ctx.Tensor("w").Shape().at(1)) {
        return true;
      }
      return false;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &x_num_col_dims_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> std::any { return 1; });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &fused_fc_elementwise_op =
        res.Op(paddle::dialect::FusedFcElementwiseLayernormOp::name(),
               {{
                   {"x_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
               }});
    fused_fc_elementwise_op({&res.Tensor("x"),
                             &res.Tensor("w"),
                             &res.Tensor("y"),
                             &res.Tensor("bias0"),
                             &res.Tensor("scale"),
                             &res.Tensor("bias1")},
                            {&res.Tensor("layernorm_out"),
                             &res.Tensor("layernorm_mean"),
                             &res.Tensor("layernorm_variance")});
  }
};

class FcElementwiseLayerNormFuse2Pattern
    : public pir::drr::DrrPatternBase<FcElementwiseLayerNormFuse2Pattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"use_mkldnn", pat.Attr("use_mkldnn")},
                   {"padding_weights", pat.Attr("padding_weights")},
                   {"use_quantizer", pat.Attr("use_quantizer")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"scale_in", pat.Attr("scale_in")},
                   {"scale_weights", pat.Attr("scale_weights")},
                   {"scale_out", pat.Attr("scale_out")},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
               });
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias0")},
       {&pat.Tensor("fc_out")});
    add({&pat.Tensor("y"), &pat.Tensor("fc_out")}, {&pat.Tensor("add_out")});
    layernorm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias1")},
        {&pat.Tensor("layernorm_out"),
         &pat.Tensor("layernorm_mean"),
         &pat.Tensor("layernorm_variance")});
    // Constrains the activation is none
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      int64_t layer_norm_x = 1;
      for (int i = match_ctx.Attr<int>("begin_norm_axis");
           i < match_ctx.Tensor("fc_out").Shape().size();
           i++) {
        layer_norm_x *= match_ctx.Tensor("fc_out").Shape().at(i);
      }
      if (layer_norm_x == match_ctx.Tensor("w").Shape().at(1)) {
        return true;
      }
      return false;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_fc_elementwise_op =
        res.Op(paddle::dialect::FusedFcElementwiseLayernormOp::name(),
               {{
                   {"x_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
               }});
    fused_fc_elementwise_op({&res.Tensor("x"),
                             &res.Tensor("w"),
                             &res.Tensor("y"),
                             &res.Tensor("bias0"),
                             &res.Tensor("scale"),
                             &res.Tensor("bias1")},
                            {&res.Tensor("layernorm_out"),
                             &res.Tensor("layernorm_mean"),
                             &res.Tensor("layernorm_variance")});
  }
};

class FcElementwiseLayerNormFusePass : public pir::PatternRewritePass {
 public:
  FcElementwiseLayerNormFusePass()
      : pir::PatternRewritePass("fc_elementwise_layernorm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(FcElementwiseLayerNormFusePattern().Build(context));
    ps.Add(FcElementwiseLayerNormFuse2Pattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcElementwiseLayerNormFusePass() {
  return std::make_unique<FcElementwiseLayerNormFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_elementwise_layernorm_fuse_pass,
                 FcElementwiseLayerNormFusePass);
