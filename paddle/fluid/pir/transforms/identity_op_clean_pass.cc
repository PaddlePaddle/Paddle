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

#include "paddle/fluid/pir/transforms/identity_op_clean_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class RemoveUselessScalePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveUselessScalePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &scale_op =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    scale_op({&pat.Tensor("x"), &full_op()}, {&pat.Tensor("scale_out")});

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<float>("value") == 1.0 &&
              match_ctx.Attr<float>("bias") == 0.0);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("scale_out").Assign(res.Tensor("x"));
  }
};

class RemoveRedundantScalePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveRedundantScalePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_op_1 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_1")},
                                    {"value", pat.Attr("value_1")},
                                    {"dtype", pat.Attr("dtype_1")},
                                    {"place", pat.Attr("place_1")}});
    const auto &scale_op_1 =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias_1")},
                {"bias_after_scale", pat.Attr("bias_after_scale_1")}});
    const auto &full_op_2 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_2")},
                                    {"value", pat.Attr("value_2")},
                                    {"dtype", pat.Attr("dtype_2")},
                                    {"place", pat.Attr("place_2")}});
    const auto &scale_op_2 =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias_2")},
                {"bias_after_scale", pat.Attr("bias_after_scale_2")}});
    scale_op_1({&pat.Tensor("x"), &full_op_1()}, {&pat.Tensor("scale_1_out")});
    scale_op_2({&pat.Tensor("scale_1_out"), &full_op_2()},
               {&pat.Tensor("scale_2_out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &bias_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          float res_bias_1 = 0.f;
          float res_bias_2 = 0.f;
          if (match_ctx.Attr<bool>("bias_after_scale_1")) {
            res_bias_1 = match_ctx.Attr<float>("bias_1");
          } else {
            res_bias_1 = match_ctx.Attr<float>("value_1") *
                         match_ctx.Attr<float>("bias_1");
          }
          if (match_ctx.Attr<bool>("bias_after_scale_2")) {
            res_bias_2 = res_bias_1 * match_ctx.Attr<float>("value_2") +
                         match_ctx.Attr<float>("bias_2");
          } else {
            res_bias_2 = (res_bias_1 + match_ctx.Attr<float>("bias_2")) *
                         match_ctx.Attr<float>("value_2");
          }
          return res_bias_2;
        });
    const auto &res_scale_input = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("value_1") *
                 match_ctx.Attr<float>("value_2");
        });

    const auto &full_op_res = res.Op(paddle::dialect::FullOp::name(),
                                     {{"shape", pat.Attr("shape_1")},
                                      {"value", res_scale_input},
                                      {"dtype", pat.Attr("dtype_1")},
                                      {"place", pat.Attr("place_1")}});
    const auto &scale_op_res =
        res.Op("pd_op.scale",
               {{"bias", bias_attr}, {"bias_after_scale", res.BoolAttr(true)}});
    scale_op_res({&res.Tensor("x"), &full_op_res()},
                 {&res.Tensor("scale_2_out")});
  }
};

class RemoveUselessCastPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveUselessCastPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    pat.Tensor("ret") = pat.Op("pd_op.cast")(pat.Tensor("arg0"));
    pat.RequireEqual(pat.Tensor("ret").dtype(), pat.Tensor("arg0").dtype());
    auto res = pat.ResultPattern();
    res.Tensor("ret").Assign(res.Tensor("arg0"));
  }
};

class RemoveUselessConcatPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveUselessConcatPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    const auto &combine = pat.Op(pir::CombineOp::name());
    combine({&pat.Tensor("x")}, {&pat.Tensor("combine_out")});
    pat.Tensor("out") = pat.Op(paddle::dialect::ConcatOp::name())(
        pat.Tensor("combine_out"), pat.Tensor("axis"));
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto combine_out = match_ctx.Tensor("combine_out");
      return combine_out.type().isa<pir::VectorType>() &&
             combine_out.type().dyn_cast<pir::VectorType>().size() == 1;
    });
    auto res = pat.ResultPattern();
    res.Tensor("out").Assign(res.Tensor("x"));
  }
};

class RemoveRedundantCastPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RemoveRedundantCastPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    pat.Tensor("tmp") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype1")}})(pat.Tensor("arg0"));
    pat.Tensor("ret") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &cast1_out_type = match_ctx.Attr<phi::DataType>("dtype1");
      return cast1_out_type != phi::DataType::INT64 &&
             cast1_out_type != phi::DataType::INT32 &&
             cast1_out_type != phi::DataType::BOOL;
    });
    auto res = pat.ResultPattern();
    res.Tensor("ret") = res.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};

class DeleteDropoutOpPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "DeleteDropoutOpPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    const auto &dropout_op =
        pat.Op("pd_op.dropout",
               {{"is_test", pat.Attr("is_test")}, {"mode", pat.Attr("mode")}});
    dropout_op({&pat.Tensor("dropout_in"), &pat.Tensor("none")},
               {&pat.Tensor("dropout_out"), &pat.Tensor("dropout_mask")});
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto is_test = match_ctx.Attr<bool>("is_test");
      auto mode = match_ctx.Attr<std::string>("mode");
      return is_test && mode == "upscale_in_train";
    });
    auto res = pat.ResultPattern();
    res.Tensor("dropout_out").Assign(res.Tensor("dropout_in"));
  }
};

class ReplaceDropoutWithScalePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "ReplaceDropoutWithScalePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    const auto &dropout_op = pat.Op("pd_op.dropout",
                                    {{"p", pat.Attr("p")},
                                     {"is_test", pat.Attr("is_test")},
                                     {"mode", pat.Attr("mode")}});
    dropout_op({&pat.Tensor("dropout_in"), &pat.Tensor("none")},
               {&pat.Tensor("dropout_out"), &pat.Tensor("dropout_mask")});
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto is_test = match_ctx.Attr<bool>("is_test");
      auto mode = match_ctx.Attr<std::string>("mode");
      return is_test && mode != "upscale_in_train";
    });

    auto res = pat.ResultPattern();

    const auto &res_scale_input = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return 1.f - match_ctx.Attr<float>("p");
        });

    const auto &full_op_res = res.Op(
        paddle::dialect::FullOp::name(),
        {{"shape",
          res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx)
                              -> phi::IntArray { return {1}; })},
         {"value", res_scale_input},
         {"dtype",
          res.ComputeAttr(
              [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
                return phi::DataType::FLOAT32;
              })},
         {"place",
          res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx)
                              -> phi::Place { return phi::CPUPlace{}; })}});
    const auto &scale_op_res =
        res.Op("pd_op.scale",
               {{"bias", res.Float32Attr(0)},
                {"bias_after_scale", res.BoolAttr(true)}});
    scale_op_res({&res.Tensor("dropout_in"), &full_op_res()},
                 {&res.Tensor("dropout_out")});
  }
};

class RemoveRedundantTransposePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "RemoveRedundantTransposePattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &transpose1 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_1")}});
    const auto &transpose2 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_2")}});

    pat.Tensor("ret") = transpose2(transpose1(pat.Tensor("arg_transpose")));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &new_perm_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          const auto &perm1 = match_ctx.Attr<std::vector<int>>("perm_1");
          const auto &perm2 = match_ctx.Attr<std::vector<int>>("perm_2");
          std::vector<int> new_perm;
          for (int v : perm2) {
            new_perm.emplace_back(perm1[v]);
          }
          return new_perm;
        });
    const auto &tranpose_continuous =
        res.Op("pd_op.transpose", {{"perm", new_perm_attr}});

    res.Tensor("ret") = tranpose_continuous(res.Tensor("arg_transpose"));
  }
};

class IdentityOpCleanPass : public pir::PatternRewritePass {
 public:
  IdentityOpCleanPass()
      : pir::PatternRewritePass("identity_op_clean_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<RemoveUselessScalePattern>(context));
    ps.Add(paddle::drr::Create<RemoveRedundantScalePattern>(context));
    ps.Add(paddle::drr::Create<RemoveUselessCastPattern>(context));
    ps.Add(paddle::drr::Create<RemoveUselessConcatPattern>(context));
    ps.Add(paddle::drr::Create<RemoveRedundantCastPattern>(context));
    ps.Add(paddle::drr::Create<DeleteDropoutOpPattern>(context));
    ps.Add(paddle::drr::Create<ReplaceDropoutWithScalePattern>(context));
    ps.Add(paddle::drr::Create<RemoveRedundantTransposePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateIdentityOpCleanPass() {
  return std::make_unique<IdentityOpCleanPass>();
}
}  // namespace pir

REGISTER_IR_PASS(identity_op_clean_pass, IdentityOpCleanPass);
