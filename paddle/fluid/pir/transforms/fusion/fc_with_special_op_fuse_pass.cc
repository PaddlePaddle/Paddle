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

#include "paddle/fluid/pir/transforms/fusion/fc_with_special_op_fuse_pass.h"

#include "paddle/common/ddim.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class SqueezeFcFusePattern
    : public pir::drr::DrrPatternBase<SqueezeFcFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &squeeze_op = pat.Op(paddle::dialect::SqueezeOp::name());
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    squeeze_op({&pat.Tensor("x"), &pat.Tensor("axis")},
               {&pat.Tensor("squeeze_out"), &pat.Tensor("xshape")});
    matmul({&pat.Tensor("squeeze_out"), &pat.Tensor("w")},
           {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = add(pat.Tensor("matmul_out"), pat.Tensor("bias"));
    // Constrains the activation is none
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      auto axis_type = match_ctx.Tensor("axis").Dtype().get();
      if (axis_type.isa<pir::VectorType>() &&
          axis_type.dyn_cast<pir::VectorType>().size() != 2) {
        return false;
      }

      if (!axis_type.isa<pir::VectorType>() &&
          match_ctx.Tensor("axis").Shape().size() > 0 &&
          match_ctx.Tensor("axis").Shape().at(0) != 2) {
        return false;
      }

      if (match_ctx.Tensor("x").Shape().size() != 4 ||
          match_ctx.Tensor("x").Shape().at(2) != 1 ||
          match_ctx.Tensor("x").Shape().at(3) != 1 ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }

      if (match_ctx.Tensor("w").Shape().size() != 2 ||
          match_ctx.Tensor("squeeze_out").Shape().size() != 2) {
        return false;
      }
      if (match_ctx.Tensor("squeeze_out").Shape().at(1) !=
          match_ctx.Tensor("w").Shape().at(0)) {
        return false;
      }
      if (match_ctx.Tensor("bias").Shape().size() == 1) {
        return match_ctx.Tensor("bias").Shape().at(0) ==
               match_ctx.Tensor("w").Shape().at(1);
      }
      if (match_ctx.Tensor("bias").Shape().size() == 2) {
        return match_ctx.Tensor("bias").Shape().at(0) == 1 &&
               match_ctx.Tensor("bias").Shape().at(1) ==
                   match_ctx.Tensor("w").Shape().at(1);
      }
      return false;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> std::any { return 1; });
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
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
       {&res.Tensor("add_out")});
  }
};

class ReshapeFcFusePattern
    : public pir::drr::DrrPatternBase<ReshapeFcFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &reshape_op = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    reshape_op({&pat.Tensor("x"), &pat.Tensor("shape")},
               {&pat.Tensor("reshape_out"), &pat.Tensor("xshape")});
    matmul({&pat.Tensor("reshape_out"), &pat.Tensor("w")},
           {&pat.Tensor("matmul_out")});
    add({&pat.Tensor("matmul_out"), &pat.Tensor("bias")},
        {&pat.Tensor("add_out")});
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      if (match_ctx.Tensor("w").Shape().size() != 2 ||
          match_ctx.Attr<bool>("transpose_x") == true ||
          match_ctx.Attr<bool>("transpose_y") == true) {
        return false;
      }
      if (match_ctx.Tensor("reshape_out").Shape().size() < 2 ||
          (match_ctx.Tensor("reshape_out").Shape().size() > 0 &&
           match_ctx.Tensor("reshape_out")
                   .Shape()
                   .at(match_ctx.Tensor("reshape_out").Shape().size() - 1) !=
               match_ctx.Tensor("w").Shape().at(0))) {
        return false;
      }

      if (match_ctx.Tensor("bias").Shape().size() == 1 &&
          match_ctx.Tensor("bias").Shape().at(0) !=
              match_ctx.Tensor("w").Shape().at(1)) {
        return false;
      }
      if (match_ctx.Tensor("bias").Shape().size() == 2 &&
          (match_ctx.Tensor("bias").Shape().at(0) != 1 ||
           match_ctx.Tensor("bias").Shape().at(1) !=
               match_ctx.Tensor("w").Shape().at(1))) {
        return false;
      }

      if (match_ctx.Tensor("x").Shape().size() <
          match_ctx.Tensor("reshape_out").Shape().size()) {
        return false;
      }
      int i = match_ctx.Tensor("x").Shape().size() - 1;
      int j = match_ctx.Tensor("reshape_out").Shape().size() - 1;
      int target = match_ctx.Tensor("reshape_out").Shape().at(j);
      int mul = match_ctx.Tensor("x").Shape().at(i);

      if (mul > target) {
        return false;
      }
      /*
      reshape_in:[2,12,12,128]
      reshape_out:[2,144,128]
      shape:[2,144,128]
      n = len(reshape_in)
      m = len(reshape_out)

      request:
      1. reshape_in[i:] = reshape_in[i]*reshape_in[i+1]...*reshape_in[n-1]
      reshape_in[i:]=reshape_out=[j]
      2.reshape_in[:i]=reshape_out[:j]
      e.g.:
                                        288(reshape_out[:2])
                i    shape[2,144,128]   |  |  j(invariable)
      [2,12,12,128]------------------->[2,144,128]
       |    |   |                              |
       |    |   |reshape_in[i]=reshape_out[j]  |
       \    /
        288(reshape_in[:3])


      */
      while (target != mul) {
        if (mul <= 0 || mul > target) {
          return false;
        }
        i--;
        mul *= match_ctx.Tensor("x").Shape().at(i);
      }

      int mul1 = 1;
      int mul2 = 1;
      i--;
      j--;
      while (i >= 0 || j >= 0) {
        if (i >= 0) {
          mul1 *= match_ctx.Tensor("x").Shape().at(i);
          i--;
        }
        if (j >= 0) {
          mul2 *= match_ctx.Tensor("x").Shape().at(j);
          j--;
        }
      }
      if (mul1 != mul2) {
        return false;
      }
      return true;
    });
    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          int i = match_ctx.Tensor("x").Shape().size() - 1;
          int target =
              match_ctx.Tensor("reshape_out")
                  .Shape()
                  .at(match_ctx.Tensor("reshape_out").Shape().size() - 1);
          int mul = match_ctx.Tensor("x").Shape().at(i);
          while (target != mul) {
            i--;
            mul *= match_ctx.Tensor("x").Shape().at(i);
          }
          return i;
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
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
       {&res.Tensor("add_out")});
  }
};

class FlattenFcFusePattern
    : public pir::drr::DrrPatternBase<FlattenFcFusePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &flatten_op = pat.Op(paddle::dialect::FlattenOp::name(),
                                    {{"start_axis", pat.Attr("start_axis")},
                                     {"stop_axis", pat.Attr("stop_axis")}});
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    flatten_op({&pat.Tensor("x")},
               {&pat.Tensor("flatten_out"), &pat.Tensor("xshape")});
    matmul({&pat.Tensor("flatten_out"), &pat.Tensor("w")},
           {&pat.Tensor("matmul_out")});
    pat.Tensor("add_out") = add(pat.Tensor("matmul_out"), pat.Tensor("bias"));
    // Constrains the activation is none
    pat.RequireNativeCall([&](const pir::drr::MatchContext &match_ctx) {
      bool flatten_flag = false;

      if (match_ctx.Tensor("x").Shape().size() == 4 &&
          match_ctx.Tensor("flatten_out").Shape().size() == 2 &&
          match_ctx.Attr<int>("start_axis") == 1 &&
          match_ctx.Attr<int>("stop_axis") == 3 &&
          match_ctx.Attr<bool>("transpose_x") == false &&
          match_ctx.Attr<bool>("transpose_y") == false) {
        flatten_flag = true;
      }

      if (match_ctx.Tensor("w").Shape().size() != 2 ||
          match_ctx.Tensor("flatten_out").Shape().size() != 2) {
        return false;
      }
      if (match_ctx.Tensor("flatten_out").Shape().at(1) !=
          match_ctx.Tensor("w").Shape().at(0)) {
        return false;
      }
      if (match_ctx.Tensor("bias").Shape().size() == 1) {
        return flatten_flag && match_ctx.Tensor("bias").Shape().at(0) ==
                                   match_ctx.Tensor("w").Shape().at(1);
      }
      if (match_ctx.Tensor("bias").Shape().size() == 2) {
        return flatten_flag && match_ctx.Tensor("bias").Shape().at(0) == 1 &&
               match_ctx.Tensor("bias").Shape().at(1) ==
                   match_ctx.Tensor("w").Shape().at(1);
      }
      return false;
    });

    pir::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr = res.Attr(
        [](const pir::drr::MatchContext &match_ctx) -> std::any { return 1; });
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
    fc({&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
       {&res.Tensor("add_out")});
  }
};

class FcWithSpecialOpFusePass : public pir::PatternRewritePass {
 public:
  FcWithSpecialOpFusePass()
      : pir::PatternRewritePass("fc_with_special_op_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(SqueezeFcFusePattern().Build(context));
    ps.Add(ReshapeFcFusePattern().Build(context));
    ps.Add(FlattenFcFusePattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFcWithSpecialOpFusePass() {
  return std::make_unique<FcWithSpecialOpFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_with_special_op_fuse_pass, FcWithSpecialOpFusePass);
