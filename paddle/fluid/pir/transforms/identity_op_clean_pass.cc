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
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_add_fuse_pass.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/common/ddim.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class RemoveUselessScalePattern
    : public pir::drr::DrrPatternBase<RemoveUselessScalePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
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

    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      return (match_ctx.Attr<float>("value") == 1.0 &&
              match_ctx.Attr<float>("bias") == 0.0);
    });

    pir::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("scale_out").Assign(res.Tensor("x"));
  }
};

class RemoveRedundentScalePattern
    : public pir::drr::DrrPatternBase<RemoveRedundentScalePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
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

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &bais_res =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
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
    const auto &res_scale_input =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> float {
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
               {{"bias", bais_res},
                {"bias_after_scale",
                 res.Attr([](const pir::drr::MatchContext &match_ctx) -> bool {
                   return true;
                 })}});
    scale_op_res({&res.Tensor("x"), &full_op_res()},
                 {&res.Tensor("scale_2_out")});
  }
};

class RemoveUselessCastPattern
    : public pir::drr::DrrPatternBase<RemoveUselessCastPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    pat.Tensor("ret") = pat.Op("pd_op.cast")(pat.Tensor("arg0"));
    pat.RequireEqual(pat.Tensor("ret").dtype(), pat.Tensor("arg0").dtype());
    auto res = pat.ResultPattern();
    res.Tensor("ret").Assign(res.Tensor("arg0"));
  }
};

class RemoveUselessConcatPattern
    : public pir::drr::DrrPatternBase<RemoveUselessConcatPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    const auto &combine = pat.Op(pir::CombineOp::name());
    combine({&pat.Tensor("x")}, {&pat.Tensor("combine_out")});
    pat.Tensor("out") = pat.Op(paddle::dialect::ConcatOp::name())(
        pat.Tensor("combine_out"), pat.Tensor("axis"));
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      auto x_type = dynamic_cast<const pir::drr::IrValue &>(
                        match_ctx.Tensor("combine_out"))
                        .get()
                        .type();
      return x_type.isa<pir::VectorType>() &&
             x_type.dyn_cast<pir::VectorType>().size() == 1;
    });
    auto res = pat.ResultPattern();
    res.Tensor("out").Assign(res.Tensor("x"));
  }
};

class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    auto pat = ctx->SourcePattern();
    pat.Tensor("tmp") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype1")}})(pat.Tensor("arg0"));
    pat.Tensor("ret") = pat.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    auto res = pat.ResultPattern();
    res.Tensor("ret") = res.Op(
        "pd_op.cast", {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};

class RemoveRedundentTransposePattern
    : public pir::drr::DrrPatternBase<RemoveRedundentTransposePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &transpose1 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_1")}});
    const auto &transpose2 =
        pat.Op("pd_op.transpose", {{"perm", pat.Attr("perm_2")}});

    pat.Tensor("ret") = transpose2(transpose1(pat.Tensor("arg_transpose")));

    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &new_perm_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> std::vector<int> {
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
    ps.Add(RemoveUselessScalePattern().Build(context));
    ps.Add(RemoveRedundentScalePattern().Build(context));
    ps.Add(RemoveUselessCastPattern().Build(context));
    ps.Add(RemoveUselessConcatPattern().Build(context));
    ps.Add(RemoveRedundentCastPattern().Build(context));
    ps.Add(RemoveRedundentTransposePattern().Build(context));
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
