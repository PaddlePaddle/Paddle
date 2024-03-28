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

#include "paddle/fluid/pir/transforms/gpu/add_norm_fuse_pass.h"

#include <string>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class RmsNormFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RmsNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &pow = pat.Op(paddle::dialect::PowOp::name());
    const auto &mean =
        pat.Op(paddle::dialect::MeanOp::name(), {{"axis", pat.Attr("axis")}});
    const auto &full = pat.Op(paddle::dialect::FullOp::name());
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(), {{"bias", pat.Attr("bias")}});
    const auto &rsqrt = pat.Op(paddle::dialect::RsqrtOp::name());
    const auto &multiply1 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply2 = pat.Op(paddle::dialect::MultiplyOp::name());

    pat.Tensor("pow_out") = pow(pat.Tensor("x"));
    pat.Tensor("mean_out") = mean(pat.Tensor("pow_out"));
    pat.Tensor("scale_out") = scale(pat.Tensor("mean_out"), full());
    pat.Tensor("rsqrt_out") = rsqrt(pat.Tensor("scale_out"));
    pat.Tensor("multiply_out1") =
        multiply1(pat.Tensor("rsqrt_out"), pat.Tensor("x"));
    pat.Tensor("multiply_out2") =
        multiply2(pat.Tensor("multiply_out1"), pat.Tensor("w"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto axis = match_ctx.Attr<std::vector<int64_t>>("axis");
      if (axis.size() > 1) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &begin_norm_axis =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          const auto &axis = match_ctx.Attr<std::vector<int64_t>>("axis");
          auto pow_out_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("pow_out"));
          return axis[0] == -1 ? static_cast<int>(pow_out_shape.size()) - 1
                               : axis[0];
        });

    const auto &rms_norm = res.Op(paddle::dialect::RmsNormOp::name(),
                                  {{
                                      {"epsilon", pat.Attr("bias")},
                                      {"begin_norm_axis", begin_norm_axis},
                                      {"quant_scale", res.Float32Attr(-1.0)},
                                      {"quant_round_type", res.Int32Attr(0)},
                                      {"quant_max_bound", res.Float32Attr(0.0)},
                                      {"quant_min_bound", res.Float32Attr(0.0)},
                                  }});

    rms_norm(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.Tensor("w"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("multiply_out2"),
         &res.Tensor("residual_out"),
         &res.Tensor("inv_var")});
  }
};

class AddRmsNormFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "AddRmsNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &pat_rms_norm =
        pat.Op(paddle::dialect::RmsNormOp::name(),
               {
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"quant_scale", pat.Attr("quant_scale")},
                   {"quant_round_type", pat.Attr("quant_round_type")},
                   {"quant_max_bound", pat.Attr("quant_max_bound")},
                   {"quant_min_bound", pat.Attr("quant_min_bound")},
               });
    pat.Tensor("add_out") = add(pat.Tensor("x"), pat.Tensor("residual"));
    pat_rms_norm({&pat.Tensor("add_out"),
                  &pat.InputNoneTensor(),
                  &pat.InputNoneTensor(),
                  &pat.Tensor("w"),
                  &pat.InputNoneTensor()},
                 {&pat.Tensor("rms_norm_out"),
                  &pat.Tensor("residual_out_0"),
                  &pat.Tensor("inv_var_0")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &res_rms_norm =
        res.Op(paddle::dialect::RmsNormOp::name(),
               {
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"quant_scale", pat.Attr("quant_scale")},
                   {"quant_round_type", pat.Attr("quant_round_type")},
                   {"quant_max_bound", pat.Attr("quant_max_bound")},
                   {"quant_min_bound", pat.Attr("quant_min_bound")},
               });

    res_rms_norm(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.Tensor("residual"),
            &res.Tensor("w"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("rms_norm_out"),
         &res.Tensor("residual_out"),
         &res.Tensor("inv_var")});
  }
};

class AddLayerNormFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "AddLayerNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layer_norm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    pat.Tensor("add_out") = add(pat.Tensor("x"), pat.Tensor("residual"));
    layer_norm(
        {&pat.Tensor("add_out"), &pat.Tensor("w"), &pat.InputNoneTensor()},
        {&pat.Tensor("layer_norm_out"),
         &pat.Tensor("mean_out_0"),
         &pat.Tensor("variance_out_0")});

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fuse_layer_norm =
        res.Op(paddle::dialect::FusedBiasResidualLayernormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"residual_alpha", res.Float32Attr(1.0)},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                {"quant_scale", res.Float32Attr(-1.0)},
                {"quant_round_type", res.Int32Attr(0)},
                {"quant_max_bound", res.Float32Attr(0.0)},
                {"quant_min_bound", res.Float32Attr(0.0)}});

    fuse_layer_norm(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.Tensor("residual"),
            &res.Tensor("w"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("layer_norm_out"),
         &res.Tensor("residual_out"),
         &res.Tensor("mean_out"),
         &res.Tensor("variance_out")});
  }
};

class AddNormFusePass : public pir::PatternRewritePass {
 public:
  AddNormFusePass() : pir::PatternRewritePass("add_norm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    // x-pow-mean-scale->rsqrt-
    //                          mul--
    // x-----------------------
    //                                mul --->rms_norm
    // w-----------------------------
    ps.Add(paddle::drr::Create<RmsNormFusePattern>(context));
    // x--------
    //           add-rms_norm --- rms_norm
    // residual-
    ps.Add(paddle::drr::Create<AddRmsNormFusePattern>(context));
    // x--------
    //           add-layer_norm ---- fused_bias_residual_layernorm
    // residual-
    ps.Add(paddle::drr::Create<AddLayerNormFusePattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAddNormFusePass() {
  return std::make_unique<AddNormFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(add_norm_fuse_pass, AddNormFusePass);
