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

#include "paddle/fluid/pir/transforms/fusion/fused_gemm_epilogue_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class FusedLinearPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto bias_dims = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      return (w_dims.size() == 2 && x_dims.size() >= 2 &&
              bias_dims.size() == 1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }

  std::string pattern_name() const override { return "FusedLinearPattern"; }
};

class FusedLinearGradPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &matmul_grad = pat.Op(paddle::dialect::MatmulGradOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x")},
                                      {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));
    add_grad({&pat.Tensor("tmp"), &pat.Tensor("bias"), &pat.Tensor("out_grad")},
             {&pat.Tensor("tmp_grad"), &pat.Tensor("bias_grad")});
    matmul_grad({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("tmp_grad")},
                {&pat.Tensor("x_grad"), &pat.Tensor("w_grad")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_dims = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_dims = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto bias_dims = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      return (w_dims.size() == 2 && x_dims.size() >= 2 &&
              bias_dims.size() == 1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", act_attr}}});
    const auto &fused_gemm_epilogue_grad =
        res.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation_grad", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
    fused_gemm_epilogue_grad({&res.Tensor("x"),
                              &res.Tensor("w"),
                              &res.NoneTensor(),
                              &res.Tensor("out_grad")},
                             {&res.Tensor("x_grad"),
                              &res.Tensor("w_grad"),
                              &res.Tensor("bias_grad")});
  }

  std::string pattern_name() const override { return "FusedLinearGradPattern"; }
};

class FusedLinearGeluPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    // Source pattern
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", pat.Attr("act")}}});
    const auto &gelu = pat.Op(paddle::dialect::GeluOp::name());
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") = gelu(pat.Tensor("fuse_out"));

    // Constrains the activation is none
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<std::string>("act") == "none");
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "gelu";
        });
    const auto &fused_gemm_epilogue_gelu =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", act_attr}}});
    fused_gemm_epilogue_gelu(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space")});
  }

  std::string pattern_name() const override { return "FusedLinearGeluPattern"; }
};

class FusedLinearReluPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    // Source pattern
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", pat.Attr("act")}}});
    const auto &relu = pat.Op(paddle::dialect::ReluOp::name());
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") = relu(pat.Tensor("fuse_out"));

    // Constrains the activation is none
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<std::string>("act") == "none");
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "relu";
        });
    const auto &fused_gemm_epilogue_relu =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation", act_attr}}});
    fused_gemm_epilogue_relu(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space")});
  }

  std::string pattern_name() const override { return "FusedLinearReluPattern"; }
};

class FusedLinearGeluGradPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", pat.Attr("act1")}}});
    const auto &fused_gemm_epilogue_grad1 =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", pat.Attr("act2")}}});
    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    pat.Tensor("out") =
        pat.Op(paddle::dialect::GeluOp::name())(pat.Tensor("fuse_out"));

    fused_gemm_epilogue_grad1({&pat.Tensor("x1"),
                               &pat.Tensor("w1"),
                               &pat.Tensor("reserve_space1"),
                               &pat.Tensor("out_grad")},
                              {&pat.Tensor("x1_grad"),
                               &pat.Tensor("w1_grad"),
                               &pat.Tensor("bias1_grad")});
    pat.Tensor("gelu_dx") = pat.Op(paddle::dialect::GeluGradOp::name())(
        pat.Tensor("fuse_out"), pat.Tensor("x1_grad"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("act1") == "none" &&
             match_ctx.Attr<std::string>("act2") == "none";
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "gelu";
        });
    const auto &fused_gemm_epilogue_new =
        res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", act_attr}}});
    const auto &act_grad_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "gelu_grad";
        });
    const auto &fused_gemm_epilogue_grad_new =
        res.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", act_grad_attr}}});
    fused_gemm_epilogue_new(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out"), &res.Tensor("reserve_space2")});
    fused_gemm_epilogue_grad_new({&res.Tensor("x1"),
                                  &res.Tensor("w1"),
                                  &res.Tensor("reserve_space2"),
                                  &res.Tensor("out_grad")},
                                 {&res.Tensor("gelu_dx"),
                                  &res.Tensor("w1_grad"),
                                  &res.Tensor("bias1_grad")});
  }

  std::string pattern_name() const override {
    return "FusedLinearGeluGradPattern";
  }
};

class FusedLinearReluGradPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fused_gemm_epilogue =
        pat.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
               {{{"trans_x", pat.Attr("trans_x1")},
                 {"trans_y", pat.Attr("trans_y1")},
                 {"activation", pat.Attr("act1")}}});
    const auto &fused_gemm_epilogue_grad =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x2")},
                 {"trans_y", pat.Attr("trans_y2")},
                 {"activation_grad", pat.Attr("act2")}}});
    const auto &fused_gemm_epilogue_grad1 =
        pat.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x3")},
                 {"trans_y", pat.Attr("trans_y3")},
                 {"activation_grad", pat.Attr("act3")}}});

    fused_gemm_epilogue(
        {&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("bias")},
        {&pat.Tensor("fuse_out"), &pat.Tensor("reserve_space")});
    fused_gemm_epilogue_grad1({&pat.Tensor("x1"),
                               &pat.Tensor("w1"),
                               &pat.Tensor("reserve_space2"),
                               &pat.Tensor("out_grad")},
                              {&pat.Tensor("x1_grad"),
                               &pat.Tensor("w1_grad"),
                               &pat.Tensor("bias1_grad")});

    pat.Tensor("relu_dx") = pat.Op(paddle::dialect::ReluGradOp::name())(
        pat.Tensor("x1"), pat.Tensor("x1_grad"));
    fused_gemm_epilogue_grad({&pat.Tensor("x"),
                              &pat.Tensor("w"),
                              &pat.Tensor("reserve_space1"),
                              &pat.Tensor("relu_dx")},
                             {&pat.Tensor("x_grad"),
                              &pat.Tensor("w_grad"),
                              &pat.Tensor("bias_grad")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return match_ctx.Attr<std::string>("act1") == "relu" &&
             match_ctx.Attr<std::string>("act3") == "none";
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_grad_attr =
        res.Attr([](const paddle::drr::MatchContext &match_ctx) -> std::any {
          return "relu_grad";
        });
    const auto &res_fused_gemm_epilogue_grad1 =
        res.Op(paddle::dialect::FusedGemmEpilogueGradOp::name(),
               {{{"trans_x", pat.Attr("trans_x3")},
                 {"trans_y", pat.Attr("trans_y3")},
                 {"activation_grad", act_grad_attr}}});

    res_fused_gemm_epilogue_grad1({&res.Tensor("x1"),
                                   &res.Tensor("w1"),
                                   &res.Tensor("reserve_space"),
                                   &res.Tensor("out_grad")},
                                  {&res.Tensor("relu_dx"),
                                   &res.Tensor("w1_grad"),
                                   &res.Tensor("bias1_grad")});
  }

  std::string pattern_name() const override {
    return "FusedLinearReluGradPattern";
  }
};

class FusedGemmEpiloguePass : public pir::PatternRewritePass {
 public:
  FusedGemmEpiloguePass()
      : pir::PatternRewritePass("fused_gemm_epilogue_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(FusedLinearGradPattern().Build(context));
    ps.Add(FusedLinearPattern().Build(context));
    ps.Add(FusedLinearGeluPattern().Build(context));
    ps.Add(FusedLinearReluPattern().Build(context));
    ps.Add(FusedLinearGeluGradPattern().Build(context));
    ps.Add(FusedLinearReluGradPattern().Build(context));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedGemmEpiloguePass() {
  return std::make_unique<FusedGemmEpiloguePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fused_gemm_epilogue_pass, FusedGemmEpiloguePass);
