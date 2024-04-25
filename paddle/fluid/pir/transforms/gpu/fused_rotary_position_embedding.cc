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

#include "paddle/fluid/pir/transforms/gpu/fused_rotary_position_embedding.h"

#include <string>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FusedRotaryPositionEmbeddingPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RotaryPositionEmbeddingPattern"; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &squeeze = pat.Op(paddle::dialect::SqueezeOp::name());
    const auto &squeeze_1 = pat.Op(paddle::dialect::SqueezeOp::name());

    const auto &gather_nd = pat.Op(paddle::dialect::GatherNdOp::name());
    const auto &unsqueeze = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_1 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_2 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_1 = pat.Op(paddle::dialect::AddOp::name());
    const auto &matmul_op = pat.Op(paddle::dialect::MatmulOp::name(),
                                   {{"transpose_x", pat.Attr("trans_x")},
                                    {"transpose_y", pat.Attr("trans_y")}});
    const auto &matmul_op_1 = pat.Op(paddle::dialect::MatmulOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x_1")},
                                      {"transpose_y", pat.Attr("trans_y_1")}});
    const auto &matmul_op_2 = pat.Op(paddle::dialect::MatmulOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x_2")},
                                      {"transpose_y", pat.Attr("trans_y_2")}});
    const auto &matmul_op_3 = pat.Op(paddle::dialect::MatmulOp::name(),
                                     {{"transpose_x", pat.Attr("trans_x_3")},
                                      {"transpose_y", pat.Attr("trans_y_3")}});

    const auto &slice_q =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});
    const auto &slice_q_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});

    const auto &slice_k =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});

    const auto &slice_k_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});

    const auto &full_int_array_q1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q1")}});
    const auto &full_int_array_q2 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q2")}});
    const auto &full_int_array_q3 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q3")}});
    const auto &full_int_array_q4 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q4")}});
    const auto &full_int_array_q5 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q5")}});
    const auto &full_int_array_q6 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q6")}});
    const auto &full_int_array_q7 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q7")}});
    const auto &full_int_array_q8 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"value", pat.Attr("full_int_array_q8")}});

    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &full_op_1 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_1")},
                                    {"value", pat.Attr("value_1")},
                                    {"dtype", pat.Attr("dtype_1")},
                                    {"place", pat.Attr("place_1")}});

    const auto &scale_op =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &scale_op_k =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias_q")},
                {"bias_after_scale", pat.Attr("bias_after_scale_q")}});

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &full_2 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_2_value")}});
    const auto &full_3 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_3_value")}});
    const auto &full_4 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_4_value")}});
    const auto &full_5 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_5_value")}});
    const auto &full_6 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_6_value")}});
    const auto &full_7 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_7_value")}});

    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine = pat.Op(pir::CombineOp::name());
    const auto &concat_op_k = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine_k = pat.Op(pir::CombineOp::name());

    squeeze({&pat.Tensor("cos"), &full_1()},
            {&pat.Tensor("squeeze_out_cos"), &pat.Tensor("xshape")});
    squeeze_1({&pat.Tensor("sin"), &full_2()},
              {&pat.Tensor("squeeze_out_sin"), &pat.Tensor("xshape")});

    unsqueeze({&pat.Tensor("position_ids"), &full_3()},
              {&pat.Tensor("unsqueeze_s_out"), &pat.Tensor("xshape")});
    pat.Tensor("gather_nd_out_cos") =
        gather_nd(pat.Tensor("squeeze_out_cos"), pat.Tensor("unsqueeze_s_out"));
    pat.Tensor("gather_nd_out_sin") =
        gather_nd(pat.Tensor("squeeze_out_sin"), pat.Tensor("unsqueeze_s_out"));

    unsqueeze_1({&pat.Tensor("gather_nd_out_cos"), &full_4()},
                {&pat.Tensor("unsqueeze_out_cos"), &pat.Tensor("xshape")});
    unsqueeze_2({&pat.Tensor("gather_nd_out_sin"), &full_5()},
                {&pat.Tensor("unsqueeze_out_sin"), &pat.Tensor("xshape")});
    matmul_op({&pat.Tensor("unsqueeze_out_cos"), &pat.Tensor("q")},
              {&pat.Tensor("tmp_25")});
    matmul_op_1({&pat.Tensor("unsqueeze_out_cos"), &pat.Tensor("k")},
                {&pat.Tensor("tmp_29")});

    pat.Tensor("q_slice_out1") =
        slice_q(pat.Tensor("q"), full_int_array_q1(), full_int_array_q2());
    pat.Tensor("q_slice_out2") =
        slice_q_1(pat.Tensor("q"), full_int_array_q3(), full_int_array_q4());
    scale_op({&pat.Tensor("q_slice_out2"), &full_op()},
             {{&pat.Tensor("scale_out")}});

    combine({&pat.Tensor("scale_out")}, {&pat.Tensor("combine_out")});
    concat_op({&pat.Tensor("combine_out"), &full_6()},
              {&pat.Tensor("concat_out")});
    matmul_op_2({&pat.Tensor("unsqueeze_out_sin"), &pat.Tensor("concat_out")},
                {&pat.Tensor("tmp_27")});
    pat.Tensor("tmp_28") = add(pat.Tensor("tmp_27"), pat.Tensor("tmp_25"));

    pat.Tensor("k_slice_out1") =
        slice_k(pat.Tensor("k"), full_int_array_q5(), full_int_array_q6());
    pat.Tensor("k_slice_out2") =
        slice_k_1(pat.Tensor("k"), full_int_array_q7(), full_int_array_q8());
    scale_op_k({&pat.Tensor("k_slice_out2"), &full_op_1()},
               {{&pat.Tensor("scale_out_k")}});
    combine_k({&pat.Tensor("scale_out_k")}, {&pat.Tensor("combine_out_k")});
    concat_op_k({&pat.Tensor("combine_out_k"), &full_7()},
                {&pat.Tensor("concat_out_k")});
    matmul_op_3({&pat.Tensor("unsqueeze_out_sin"), &pat.Tensor("concat_out_k")},
                {&pat.Tensor("tmp_31")});
    pat.Tensor("tmp_32") = add_1(pat.Tensor("tmp_31"), pat.Tensor("tmp_29"));
    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto check_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {0, 2};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto axis = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
      auto axis_2 = match_ctx.Attr<std::vector<int64_t>>("full_2_value");
      return check_axes(axis) && check_axes(axis_2);

      auto check_unsqueeze_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {0};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto unsqueeze_axis =
          match_ctx.Attr<std::vector<int64_t>>("full_3_value");
      auto unsqueeze_axis_1 =
          match_ctx.Attr<std::vector<int64_t>>("full_4_value");
      auto unsqueeze_axis_2 =
          match_ctx.Attr<std::vector<int64_t>>("full_5_value");
      return check_unsqueeze_axes(unsqueeze_axis) &&
             check_unsqueeze_axes(unsqueeze_axis_1) &&
             check_unsqueeze_axes(unsqueeze_axis_2);

      auto check_concat_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {-1};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto concat_axis = match_ctx.Attr<std::vector<int64_t>>("full_6_value");
      auto concat_axis_1 = match_ctx.Attr<std::vector<int64_t>>("full_7_value");
      return check_concat_axes(concat_axis) && check_concat_axes(concat_axis_1);
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_rotary_position_embedding =
        res.Op(paddle::dialect::FusedRotaryPositionEmbeddingOp::name(),
               {{
                   {"use_neox_rotary_stype", res.BoolAttr(true)},
                   {"time_major", res.BoolAttr(false)},
                   {"rotary_emb_base", res.Float32Attr(10000.0)},
               }});
    fused_rotary_position_embedding(
        {&res.Tensor("q"),
         &res.Tensor("k"),
         &res.Tensor("v"),
         &res.Tensor("sin"),
         &res.Tensor("cos"),
         &res.Tensor("position_ids")},
        {&res.Tensor("out_q"), &res.Tensor("out_k"), &res.Tensor("out_v")});
  }
};
class FusedRotaryPositionEmbeddingPass : public pir::PatternRewritePass {
 public:
  FusedRotaryPositionEmbeddingPass()
      : pir::PatternRewritePass("fused_rotary_position_embedding_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedRotaryPositionEmbeddingPattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedRotaryPositionEmbeddingPass() {
  return std::make_unique<FusedRotaryPositionEmbeddingPass>();
}
}  // namespace pir
REGISTER_IR_PASS(fused_rotary_position_embedding_pass,
                 FusedRotaryPositionEmbeddingPass);
