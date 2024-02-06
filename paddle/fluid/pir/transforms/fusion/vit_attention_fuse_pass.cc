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

#include "paddle/fluid/pir/transforms/fusion/vit_attention_fuse_pass.h"

#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class VitAttentionFusePattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_1 = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x_1")},
                                   {"transpose_y", pat.Attr("transpose_y_1")}});
    const auto &matmul_2 = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x_2")},
                                   {"transpose_y", pat.Attr("transpose_y_2")}});
    const auto &matmul_3 = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x_3")},
                                   {"transpose_y", pat.Attr("transpose_y_3")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &full_int_array_1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_1")}});
    const auto &reshape_1 = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &full_int_array_2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_2")}});
    const auto &reshape_2 = pat.Op(paddle::dialect::ReshapeOp::name());
    const auto &transpose_1 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_1")}});
    const auto &transpose_2 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_2")}});

    const auto &transpose_3 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_3")}});
    const auto &full_int_array_3 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_3")}});
    const auto &full_int_array_4 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_4")}});
    const auto &slice_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"infer_flags", pat.Attr("infer_flags_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});
    const auto &full_int_array_5 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_5")}});
    const auto &full_int_array_6 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_6")}});
    const auto &slice_2 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"infer_flags", pat.Attr("infer_flags_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});
    const auto &full_int_array_7 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_7")}});
    const auto &full_int_array_8 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("full_int_array_value_8")}});
    const auto &slice_3 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_3")},
                {"infer_flags", pat.Attr("infer_flags_3")},
                {"decrease_axis", pat.Attr("decrease_axis_3")}});
    const auto &full_1 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    const auto &softmax = pat.Op(paddle::dialect::SoftmaxOp::name(),
                                 {{"axis", pat.Attr("axis")}});

    pat.Tensor("matmul_out_1") = matmul_1(pat.Tensor("x1"), pat.Tensor("w1"));
    pat.Tensor("add_1_out") =
        add(pat.Tensor("matmul_out_1"), pat.Tensor("bias"));
    reshape_1({&pat.Tensor("add_1_out"), &full_int_array_1()},
              {&pat.Tensor("reshape_1_out"), &pat.Tensor("reshape_1_xshape")});
    pat.Tensor("transpose_1_out") = transpose_1(pat.Tensor("reshape_1_out"));
    pat.Tensor("slice_out_1") = slice_1(
        pat.Tensor("transpose_1_out"), full_int_array_3(), full_int_array_4());
    pat.Tensor("slice_out_2") = slice_2(
        pat.Tensor("transpose_1_out"), full_int_array_5(), full_int_array_6());
    pat.Tensor("slice_out_3") = slice_3(
        pat.Tensor("transpose_1_out"), full_int_array_7(), full_int_array_8());

    pat.Tensor("transpose_2_out") = transpose_2(pat.Tensor("slice_out_3"));
    pat.Tensor("matmul_out_2") =
        matmul_2(pat.Tensor("slice_out_2"), pat.Tensor("transpose_2_out"));
    pat.Tensor("scale_out") = scale(pat.Tensor("matmul_out_2"), full_1());
    pat.Tensor("softmax_out") = softmax(pat.Tensor("scale_out"));
    pat.Tensor("matmul_out_3") =
        matmul_3(pat.Tensor("softmax_out"), pat.Tensor("slice_out_1"));
    pat.Tensor("transpose_3_out") = transpose_3(pat.Tensor("matmul_out_3"));
    reshape_2({&pat.Tensor("transpose_3_out"), &full_int_array_2()},
              {&pat.Tensor("reshape_2_out"), &pat.Tensor("reshape_2_xshape")});

    // Constrains the activation is none
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto softmax_axis = match_ctx.Attr<int>("axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;
      auto matmul_out_1_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_out_1"));
      if (matmul_out_1_shape.size() != 3) {
        return false;
      }
      bool matmul_1_transpose_x_1 = match_ctx.Attr<bool>("transpose_x_1");
      bool matmul_1_transpose_y_1 = match_ctx.Attr<bool>("transpose_y_1");
      if (matmul_1_transpose_x_1 || matmul_1_transpose_y_1) return false;
      bool matmul_1_transpose_x_2 = match_ctx.Attr<bool>("transpose_x_2");
      bool matmul_1_transpose_y_2 = match_ctx.Attr<bool>("transpose_y_2");
      if (matmul_1_transpose_x_2 || matmul_1_transpose_y_2) return false;
      bool matmul_1_transpose_x_3 = match_ctx.Attr<bool>("transpose_x_3");
      bool matmul_1_transpose_y_3 = match_ctx.Attr<bool>("transpose_y_3");
      if (matmul_1_transpose_x_3 || matmul_1_transpose_y_3) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &reshape_w_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto dim_0 = pir::GetShapeFromValue(match_ctx.Tensor("w1")).at(0);
          auto dim_2 = pir::GetShapeFromValue(match_ctx.Tensor("w1")).at(1) / 3;
          return std::vector<int64_t>({dim_0, 3, dim_2});
        });

    const auto &res_reshape1 = res.Op(paddle::dialect::ReshapeOp::name(),
                                      {{"shape", reshape_w_shape_attr}});
    res_reshape1({&res.Tensor("w1")},
                 {&res.Tensor("reshape_w_out"), &res.OutputNoneTensor()});
    // Bias reshape.
    const auto &reshape_b_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto dim = pir::GetShapeFromValue(match_ctx.Tensor("bias")).at(0) / 3;
          return std::vector<int64_t>({3, dim});
        });

    const auto &res_reshape2 = res.Op(paddle::dialect::ReshapeOp::name(),
                                      {{"shape", reshape_b_shape_attr}});
    res_reshape2({&res.Tensor("bias")},
                 {&res.Tensor("reshape_bias_out"), &res.OutputNoneTensor()});

    const auto &head_number =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          return pir::GetShapeFromValue(match_ctx.Tensor("softmax_out")).at(1);
        });

    const auto &alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("full_1_value");
        });

    const auto &multihead_matmul_op =
        res.Op(paddle::dialect::MultiheadMatmulOp::name(),
               {{
                   {"transpose_q", res.BoolAttr(false)},
                   {"transpose_k", res.BoolAttr(true)},
                   {"transpose_v", res.BoolAttr(false)},
                   {"alpha", alpha},
                   {"head_number", head_number},
               }});
    multihead_matmul_op({&res.Tensor("x1"),
                         &res.Tensor("reshape_w_out"),
                         &res.Tensor("reshape_bias_out"),
                         &res.InputNoneTensor()},
                        {&res.Tensor("reshape_2_out")});
  }
  std::string name() const override { return "VitAttentionFusePattern"; }
};

class VitAttentionFusePass : public pir::PatternRewritePass {
 public:
  VitAttentionFusePass()
      : pir::PatternRewritePass("vit_attention_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<VitAttentionFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateVitAttentionFusePass() {
  return std::make_unique<VitAttentionFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(vit_attention_fuse_pass, VitAttentionFusePass);
