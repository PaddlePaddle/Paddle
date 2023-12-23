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

#include "paddle/fluid/pir/transforms/fusion/fused_linear_param_grad_add_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
namespace {

// add_grad + matmul_grad + add_ -> matmul + fused_liner_param_gard_add
class FusedMatmulAddGradAddPattern
    : public pir::drr::DrrPatternBase<FusedMatmulAddGradAddPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul0 = pat.Op(paddle::dialect::MatmulOp::name(),
                                 {{"transpose_x", pat.Attr("trans_x")},
                                  {"transpose_y", pat.Attr("trans_y")}});
    const auto &add0 = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());
    const auto &matmul_grad = pat.Op(paddle::dialect::MatmulGradOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x")},
                                      {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    pat.Tensor("out") = matmul0(pat.Tensor("x"), pat.Tensor("weight"));
    pat.Tensor("fwd_add_out") = add0(pat.Tensor("out"), pat.Tensor("bias"));
    add_grad({&pat.Tensor("out"),
              &pat.Tensor("bias"),
              &pat.Tensor("fwd_add_out_grad")},
             {&pat.Tensor("out_grad"), &pat.Tensor("dbias")});
    matmul_grad(
        {&pat.Tensor("x"), &pat.Tensor("weight"), &pat.Tensor("out_grad")},
        {&pat.Tensor("x_grad"), &pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      return (match_ctx.Tensor("weight_grad").Shape() ==
                  match_ctx.Tensor("dweight").Shape() &&
              match_ctx.Tensor("out").Shape() ==
                  match_ctx.Tensor("fwd_add_out_grad").Shape() &&
              x_trans == false && y_trans == false);
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });

    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &matmul =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", false_attr}, {"transpose_y", true_attr}});
    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", true_attr}}});

    matmul({&res.Tensor("fwd_add_out_grad"), &res.Tensor("weight")},
           {&res.Tensor("x_grad")});
    fused_linear_param_grad_add({&res.Tensor("x"),
                                 &res.Tensor("fwd_add_out_grad"),
                                 &res.Tensor("dweight"),
                                 &res.NoneTensor()},
                                {&res.Tensor("add_out"), &res.Tensor("dbias")});
  }
};

// matmul_grad + add_ -> matmul + fused_liner_param_gard_add
class FusedMatmulGradAddPattern
    : public pir::drr::DrrPatternBase<FusedMatmulGradAddPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_grad = pat.Op(paddle::dialect::MatmulGradOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x")},
                                      {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul_grad(
        {&pat.Tensor("x"), &pat.Tensor("weight"), &pat.Tensor("out_grad")},
        {&pat.Tensor("x_grad"), &pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      return (match_ctx.Tensor("weight_grad").Shape() ==
                  match_ctx.Tensor("dweight").Shape() &&
              x_trans == false && y_trans == false);
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });

    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &matmul =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", false_attr}, {"transpose_y", true_attr}});
    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", false_attr}}});

    matmul({&res.Tensor("out_grad"), &res.Tensor("weight")},
           {&res.Tensor("x_grad")});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.NoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// matmul + 0 = add_(0,1) -> fused_liner_param_gard_add
class FusedMatmulAddaPattern
    : public pir::drr::DrrPatternBase<FusedMatmulAddaPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul({&pat.Tensor("x"), &pat.Tensor("out_grad")},
           {&pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return (match_ctx.Tensor("weight_grad").Shape() ==
              match_ctx.Tensor("dweight").Shape());
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });

    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", false_attr}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.NoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// matmul + 1 = add_(1,0) -> fused_liner_param_gard_add
class FusedMatmulAddbPattern
    : public pir::drr::DrrPatternBase<FusedMatmulAddbPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul({&pat.Tensor("x"), &pat.Tensor("out_grad")},
           {&pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("weight_grad"), pat.Tensor("dweight"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return (match_ctx.Tensor("weight_grad").Shape() ==
              match_ctx.Tensor("dweight").Shape());
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });

    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &false_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return false; });

    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", false_attr}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.NoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// add_grad + matmul + 0 = add_(0,1) -> fused_liner_param_gard_add
class FusedMatmulAddGradAddaPattern
    : public pir::drr::DrrPatternBase<FusedMatmulAddGradAddaPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());
    const auto &matmul_g0 = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("trans_xg0")},
                                    {"transpose_y", pat.Attr("trans_yg0")}});
    const auto &matmul_g1 = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("trans_xg1")},
                                    {"transpose_y", pat.Attr("trans_yg1")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    pat.Tensor("out") = matmul(pat.Tensor("x"), pat.Tensor("weight"));
    pat.Tensor("fwd_add_out") = add(pat.Tensor("out"), pat.Tensor("bias"));
    add_grad({&pat.Tensor("out"), &pat.Tensor("bias"), &pat.Tensor("dadd_out")},
             {&pat.Tensor("dout"), &pat.Tensor("dbias")});
    pat.Tensor("dx") = matmul_g0(pat.Tensor("dout"), pat.Tensor("weight"));
    pat.Tensor("weight_grad") = matmul_g1(pat.Tensor("x"), pat.Tensor("dout"));
    pat.Tensor("dweight_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return (match_ctx.Tensor("weight_grad").Shape() ==
                  match_ctx.Tensor("dweight").Shape() &&
              match_ctx.Tensor("out").Shape() ==
                  match_ctx.Tensor("dadd_out").Shape());
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });
    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", true_attr}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("dadd_out"),
         &res.Tensor("dweight"),
         &res.NoneTensor()},
        {&res.Tensor("dweight_out"), &res.Tensor("dbias")});
  }
};

// add_grad + matmul + 1 = add_(1,0) -> fused_liner_param_gard_add
class FusedMatmulAddGradAddbPattern
    : public pir::drr::DrrPatternBase<FusedMatmulAddGradAddbPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());
    const auto &matmul_g0 = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("trans_xg0")},
                                    {"transpose_y", pat.Attr("trans_yg0")}});
    const auto &matmul_g1 = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("trans_xg1")},
                                    {"transpose_y", pat.Attr("trans_yg1")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    pat.Tensor("out") = matmul(pat.Tensor("x"), pat.Tensor("weight"));
    pat.Tensor("fwd_add_out") = add(pat.Tensor("out"), pat.Tensor("bias"));
    add_grad({&pat.Tensor("out"), &pat.Tensor("bias"), &pat.Tensor("dadd_out")},
             {&pat.Tensor("dout"), &pat.Tensor("dbias")});
    pat.Tensor("dx") = matmul_g0(pat.Tensor("dout"), pat.Tensor("weight"));
    pat.Tensor("weight_grad") = matmul_g1(pat.Tensor("x"), pat.Tensor("dout"));
    pat.Tensor("dweight_out") =
        add_(pat.Tensor("weight_grad"), pat.Tensor("dweight"));

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return (match_ctx.Tensor("weight_grad").Shape() ==
                  match_ctx.Tensor("dweight").Shape() &&
              match_ctx.Tensor("out").Shape() ==
                  match_ctx.Tensor("dadd_out").Shape());
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &muti_precision_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
          return !(match_ctx.Tensor("dweight").Dtype() ==
                   match_ctx.Tensor("weight_grad").Dtype());
        });
    const auto &true_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> bool { return true; });
    const auto &fused_linear_param_grad_add = res.Op(
        paddle::dialect::FusedLinearParamGradAddOp::name(),
        {{{"multi_precision", muti_precision_attr}, {"has_bias", true_attr}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("dadd_out"),
         &res.Tensor("dweight"),
         &res.NoneTensor()},
        {&res.Tensor("dweight_out"), &res.Tensor("dbias")});
  }
};

class FusedLinearParamGradAddPass : public pir::PatternRewritePass {
 public:
  FusedLinearParamGradAddPass()
      : pir::PatternRewritePass("fused_linear_param_grad_add_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(FusedMatmulAddGradAddPattern().Build(context));
    ps.Add(FusedMatmulGradAddPattern().Build(context));
    ps.Add(FusedMatmulAddaPattern().Build(context));
    ps.Add(FusedMatmulAddbPattern().Build(context));
    ps.Add(FusedMatmulAddGradAddaPattern().Build(context));
    ps.Add(FusedMatmulAddGradAddbPattern().Build(context));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedLinearParamGradAddPass() {
  return std::make_unique<FusedLinearParamGradAddPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fused_linear_param_grad_add_pass, FusedLinearParamGradAddPass);
