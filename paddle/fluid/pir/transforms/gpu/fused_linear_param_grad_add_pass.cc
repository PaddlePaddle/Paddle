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

#include "paddle/fluid/pir/transforms/gpu/fused_linear_param_grad_add_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// add_grad + matmul_grad + add_ -> matmul + fused_linear_param_grad_add
class FusedMatmulAddGradAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulAddGradAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
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

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      auto out_dims = pir::GetShapeFromValue(match_ctx.Tensor("out"));
      auto fwd_add_out_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("fwd_add_out_grad"));
      return (weight_grad_dims == dweight_dims &&
              out_dims == fwd_add_out_grad_dims && x_trans == false &&
              y_trans == false);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });

    const auto &matmul = res.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", res.BoolAttr(false)},
                                 {"transpose_y", res.BoolAttr(true)}});
    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(true)}}});

    matmul({&res.Tensor("fwd_add_out_grad"), &res.Tensor("weight")},
           {&res.Tensor("x_grad")});
    fused_linear_param_grad_add({&res.Tensor("x"),
                                 &res.Tensor("fwd_add_out_grad"),
                                 &res.Tensor("dweight"),
                                 &res.InputNoneTensor()},
                                {&res.Tensor("add_out"), &res.Tensor("dbias")});
  }
};

// matmul_grad + add_ -> matmul + fused_linear_param_grad_add
class FusedMatmulGradAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulGradAddPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_grad = pat.Op(paddle::dialect::MatmulGradOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x")},
                                      {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul_grad(
        {&pat.Tensor("x"), &pat.Tensor("weight"), &pat.Tensor("out_grad")},
        {&pat.Tensor("x_grad"), &pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      auto weight_grad_use_count = match_ctx.Tensor("weight_grad").use_count();
      return (weight_grad_dims == dweight_dims && x_trans == false &&
              y_trans == false && weight_grad_use_count == 1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });

    const auto &matmul = res.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", res.BoolAttr(false)},
                                 {"transpose_y", res.BoolAttr(true)}});
    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(false)}}});

    matmul({&res.Tensor("out_grad"), &res.Tensor("weight")},
           {&res.Tensor("x_grad")});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// matmul + reshape + reshape + matmul + reshape + add_ -> matmul +
// fused_linear_param_grad_add
class FusedMatmulReshapeMatmulAddPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedMatmulReshapeMatmulAddPattern";
  }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &reshape1 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape1({&pat.Tensor("x"), &full_int_array1()},
             {&pat.Tensor("reshape_x")});

    const auto &full_int_array2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &reshape2 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape2({&pat.Tensor("dy"), &full_int_array2()},
             {&pat.Tensor("reshape_dy")});

    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    pat.Tensor("matmul_out") =
        matmul(pat.Tensor("reshape_x"), pat.Tensor("reshape_dy"));

    const auto &full_int_array3 =
        pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &reshape3 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape3({&pat.Tensor("matmul_out"), &full_int_array3()},
             {&pat.Tensor("w_grad")});

    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());
    pat.Tensor("dweight_inplace") =
        add_(pat.Tensor("dweight"), pat.Tensor("w_grad"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      auto w_grad_dims = pir::GetShapeFromValue(match_ctx.Tensor("w_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      return (w_grad_dims == dweight_dims && x_trans == true &&
              y_trans == false);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("w_grad")));
        });

    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(false)}}});

    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("dy"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("dweight_inplace"), &res.Tensor("dbias_out")});
  }
};

// matmul + 0 = add_(0,1) -> fused_linear_param_grad_add
class FusedMatmulAddaPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulAddaPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul({&pat.Tensor("x"), &pat.Tensor("out_grad")},
           {&pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("dweight"), pat.Tensor("weight_grad"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      return (weight_grad_dims == dweight_dims);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });

    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(false)}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// matmul + 1 = add_(1,0) -> fused_linear_param_grad_add
class FusedMatmulAddbPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulAddbPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());

    matmul({&pat.Tensor("x"), &pat.Tensor("out_grad")},
           {&pat.Tensor("weight_grad")});
    pat.Tensor("add_out") =
        add_(pat.Tensor("weight_grad"), pat.Tensor("dweight"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      return (weight_grad_dims == dweight_dims);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });

    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(false)}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("out_grad"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("add_out"), &res.Tensor("dbias_out")});
  }
};

// add_grad + matmul + 0 = add_(0,1) -> fused_linear_param_grad_add
class FusedMatmulAddGradAddaPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulAddGradAddaPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
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

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      auto out_dims = pir::GetShapeFromValue(match_ctx.Tensor("out"));
      auto dadd_out_dims = pir::GetShapeFromValue(match_ctx.Tensor("dadd_out"));
      return (weight_grad_dims == dweight_dims && out_dims == dadd_out_dims);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });

    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(true)}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("dadd_out"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("dweight_out"), &res.Tensor("dbias")});
  }
};

// add_grad + matmul + 1 = add_(1,0) -> fused_linear_param_grad_add
class FusedMatmulAddGradAddbPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedMatmulAddGradAddbPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
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

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto weight_grad_dims =
          pir::GetShapeFromValue(match_ctx.Tensor("weight_grad"));
      auto dweight_dims = pir::GetShapeFromValue(match_ctx.Tensor("dweight"));
      auto out_dims = pir::GetShapeFromValue(match_ctx.Tensor("out"));
      auto dadd_out_dims = pir::GetShapeFromValue(match_ctx.Tensor("dadd_out"));
      return (weight_grad_dims == dweight_dims && out_dims == dadd_out_dims);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &multi_precision_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return !(pir::GetDataTypeFromValue(match_ctx.Tensor("dweight")) ==
                   pir::GetDataTypeFromValue(match_ctx.Tensor("weight_grad")));
        });
    const auto &fused_linear_param_grad_add =
        res.Op(paddle::dialect::FusedLinearParamGradAddOp::name(),
               {{{"multi_precision", multi_precision_attr},
                 {"has_bias", res.BoolAttr(true)}}});
    fused_linear_param_grad_add(
        {&res.Tensor("x"),
         &res.Tensor("dadd_out"),
         &res.Tensor("dweight"),
         &res.InputNoneTensor()},
        {&res.Tensor("dweight_out"), &res.Tensor("dbias")});
  }
};
class FusedLinearParamGradAddPass : public pir::PatternRewritePass {
 public:
  FusedLinearParamGradAddPass()
      : pir::PatternRewritePass("fused_linear_param_grad_add_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedMatmulAddGradAddPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulGradAddPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulAddaPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulAddbPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulAddGradAddaPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulAddGradAddbPattern>(context));
    ps.Add(paddle::drr::Create<FusedMatmulReshapeMatmulAddPattern>(context));

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
