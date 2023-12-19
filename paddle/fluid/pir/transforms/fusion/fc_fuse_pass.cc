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

#include "paddle/fluid/pir/transforms/fusion/fc_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class MatmulAddPattern : public pir::drr::DrrPatternBase<MatmulAddPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = add(pat.Tensor("matmul_out"), pat.Tensor("y"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      if (match_ctx.Tensor("w").Shape().size() != 2 ||
          match_ctx.Tensor("x").Shape().size() < 2) {
        return false;
      }
      if (match_ctx.Tensor("x").Shape().at(
              match_ctx.Tensor("x").Shape().size() - 1) !=
              match_ctx.Tensor("w").Shape().at(0) ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }
      if (match_ctx.Tensor("y").Shape().size() == 1) {
        return match_ctx.Tensor("y").Shape().at(0) ==
               match_ctx.Tensor("w").Shape().at(1);
      }
      if (match_ctx.Tensor("y").Shape().size() == 2) {
        return match_ctx.Tensor("y").Shape().at(0) == 1 &&
               match_ctx.Tensor("y").Shape().at(1) ==
                   match_ctx.Tensor("w").Shape().at(1);
      }
      return false;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return match_ctx.Tensor("x").Shape().size() - 1;
        });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &fc = res.Op(
        paddle::dialect::FcOp::name(),
        {{
            {"in_num_col_dims", in_num_col_dims_attr},
            {"activation_type",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::string { return ""; })},
            {"use_mkldnn", false_attr},
            {"padding_weights", false_attr},
            {"use_quantizer", false_attr},
            {"mkldnn_data_type",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::string { return "float32"; })},
            {"scale_in",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
               return 1.0f;
             })},
            {"scale_weights",
             res.Attr([](const pir::drr::MatchContext &match_ctx)
                          -> std::vector<float> { return {1.0f}; })},
            {"scale_out",
             res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
               return 1.0f;
             })},
            {"force_fp32_output", false_attr},
        }});
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
       {&res.Tensor("add_out")});
  }
};

class FcWithReluPattern : public pir::drr::DrrPatternBase<FcWithReluPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fc =
        pat.Op(paddle::dialect::FcOp::name(),
               {{
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
               }});
    fc({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("y")},
       {&pat.Tensor("fc_out")});
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());
    relu({&pat.Tensor("fc_out")}, {&pat.Tensor("relu_out")});

    // Constrains the activation is none
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("activation_type").empty();
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &fc_with_relu =
        res.Op(paddle::dialect::FcOp::name(),
               {{
                   {"in_num_col_dims", pat.Attr("in_num_col_dims")},
                   {"activation_type",
                    res.Attr([](const pir::drr::MatchContext &match_ctx)
                                 -> std::string { return "relu"; })},
                   {"use_mkldnn", pat.Attr("use_mkldnn")},
                   {"padding_weights", pat.Attr("padding_weights")},
                   {"use_quantizer", pat.Attr("use_quantizer")},
                   {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                   {"scale_in", pat.Attr("scale_in")},
                   {"scale_weights", pat.Attr("scale_weights")},
                   {"scale_out", pat.Attr("scale_out")},
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
               }});
    fc_with_relu({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("y")},
                 {&res.Tensor("relu_out")});
  }
};

class FcFusePass : public pir::PatternRewritePass {
 public:
  FcFusePass() : pir::PatternRewritePass("fc_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(MatmulAddPattern().Build(context));
    ps.Add(FcWithReluPattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcFusePass() {
  return std::make_unique<FcFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_fuse_pass, FcFusePass);
