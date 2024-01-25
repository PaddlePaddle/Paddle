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
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class FcElementwiseLayerNormFusePattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
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
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      int64_t layer_norm_x = 1;
      auto fc_out_dims = pir::GetShapeFromValue(match_ctx.Tensor("fc_out"));
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      for (size_t i = match_ctx.Attr<int>("begin_norm_axis");
           i < fc_out_dims.size();
           i++) {
        layer_norm_x *= fc_out_dims.at(i);
      }
      if (layer_norm_x == w_dims.at(1)) {
        return true;
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &x_num_col_dims_attr = res.Int32Attr(1);
    const auto &false_attr = res.BoolAttr(false);

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

  std::string name() const override {
    return "FcElementwiseLayerNormFusePattern";
  }
};

class FcElementwiseLayerNormFuse2Pattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type", pat.Attr("activation_type")},
                   {"padding_weights", pat.Attr("padding_weights")},
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
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      int64_t layer_norm_x = 1;
      auto fc_out_dims = pir::GetShapeFromValue(match_ctx.Tensor("fc_out"));
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      for (size_t i = match_ctx.Attr<int>("begin_norm_axis");
           i < fc_out_dims.size();
           i++) {
        layer_norm_x *= fc_out_dims.at(i);
      }
      if (layer_norm_x == w_dims.at(1)) {
        return true;
      }
      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

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

  std::string name() const override {
    return "FcElementwiseLayerNormFuse2Pattern";
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
