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

#include "paddle/fluid/pir/transforms/fusion/multihead_matmul_fuse_pass.h"

#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class MultiHeadMatmulFuseNoBiasQKPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // The first path to matmul with scale (q).
    const auto &matmul_1 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_1_transpose_x")},
                {"transpose_y", src.Attr("matmul_1_transpose_y")}});
    src.Tensor("matmul_1_out") =
        matmul_1(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_1_in_2"));
    const auto &add_1 = src.Op("pd_op.add");
    src.Tensor("add_1_out") =
        add_1(src.Tensor("matmul_1_out"), src.Tensor("add_1_in_2"));
    const auto &full_int_array_1 =
        src.Op("pd_op.full_int_array",
               {{"value", src.Attr("full_int_array_1_value")}});
    const auto &reshape_1 = src.Op("pd_op.reshape");
    reshape_1({&src.Tensor("add_1_out"), &full_int_array_1()},
              {&src.Tensor("reshape_1_out"), &src.Tensor("reshape_1_xshape")});
    const auto &transpose_1 = src.Op("pd_op.transpose");
    src.Tensor("transpose_1_out") = transpose_1(src.Tensor("reshape_1_out"));
    const auto &full_1 =
        src.Op("pd_op.full", {{"value", src.Attr("full_1_value")}});
    const auto &scale = src.Op("pd_op.scale");
    src.Tensor("scale_out") = scale(src.Tensor("transpose_1_out"), full_1());

    // The second path to matmul (k).
    const auto &matmul_2 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_2_transpose_x")},
                {"transpose_y", src.Attr("matmul_2_transpose_y")}});
    src.Tensor("matmul_2_out") =
        matmul_2(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_2_in_2"));
    const auto &add_2 = src.Op("pd_op.add");
    src.Tensor("add_2_out") =
        add_2(src.Tensor("matmul_2_out"), src.Tensor("add_2_in_2"));
    const auto &full_int_array_2 = src.Op("pd_op.full_int_array");
    const auto &reshape_2 = src.Op("pd_op.reshape");
    reshape_2({&src.Tensor("add_2_out"), &full_int_array_2()},
              {&src.Tensor("reshape_2_out"), &src.Tensor("reshape_2_xshape")});
    const auto &transpose_2 = src.Op("pd_op.transpose");
    src.Tensor("transpose_2_out") = transpose_2(src.Tensor("reshape_2_out"));

    // The third path to matmul (v).
    const auto &matmul_3 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_3_transpose_x")},
                {"transpose_y", src.Attr("matmul_3_transpose_y")}});
    src.Tensor("matmul_3_out") =
        matmul_3(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_3_in_2"));
    const auto &add_3 = src.Op("pd_op.add");
    src.Tensor("add_3_out") =
        add_3(src.Tensor("matmul_3_out"), src.Tensor("add_3_in_2"));
    const auto &full_int_array_3 = src.Op("pd_op.full_int_array");
    const auto &reshape_3 = src.Op("pd_op.reshape");
    reshape_3({&src.Tensor("add_3_out"), &full_int_array_3()},
              {&src.Tensor("reshape_3_out"), &src.Tensor("reshape_3_xshape")});
    const auto &transpose_3 = src.Op("pd_op.transpose");
    src.Tensor("transpose_3_out") = transpose_3(src.Tensor("reshape_3_out"));

    // softmax(qk)v
    const auto &matmul_4 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_4_transpose_x")},
                {"transpose_y", src.Attr("matmul_4_transpose_y")}});
    src.Tensor("matmul_4_out") =
        matmul_4(src.Tensor("scale_out"), src.Tensor("transpose_2_out"));

    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("matmul_4_out"));
    const auto &matmul_5 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_5_transpose_x")},
                {"transpose_y", src.Attr("matmul_5_transpose_y")}});
    src.Tensor("matmul_5_out") =
        matmul_5(src.Tensor("softmax_out"), src.Tensor("transpose_3_out"));
    const auto &transpose_4 = src.Op("pd_op.transpose");
    src.Tensor("transpose_4_out") = transpose_4(src.Tensor("matmul_5_out"));
    const auto &full_int_array_4 = src.Op("pd_op.full_int_array");
    const auto &reshape_4 = src.Op("pd_op.reshape");
    reshape_4({&src.Tensor("transpose_4_out"), &full_int_array_4()},
              {&src.Tensor("reshape_4_out"), &src.Tensor("reshape_4_xshape")});

    //
    // Constraints.
    //
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool matmul_1_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_1_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_1_transpose_x || matmul_1_transpose_y) return false;

      bool matmul_2_transpose_x = match_ctx.Attr<bool>("matmul_2_transpose_x");
      bool matmul_2_transpose_y = match_ctx.Attr<bool>("matmul_2_transpose_y");
      if (matmul_2_transpose_x || matmul_2_transpose_y) return false;

      bool matmul_3_transpose_x = match_ctx.Attr<bool>("matmul_3_transpose_x");
      bool matmul_3_transpose_y = match_ctx.Attr<bool>("matmul_3_transpose_y");
      if (matmul_3_transpose_x || matmul_3_transpose_y) return false;

      bool matmul_4_transpose_x = match_ctx.Attr<bool>("matmul_4_transpose_x");
      bool matmul_4_transpose_y = match_ctx.Attr<bool>("matmul_4_transpose_y");
      if (matmul_4_transpose_x || !matmul_4_transpose_y) return false;

      bool matmul_5_transpose_x = match_ctx.Attr<bool>("matmul_5_transpose_x");
      bool matmul_5_transpose_y = match_ctx.Attr<bool>("matmul_5_transpose_y");
      if (matmul_5_transpose_x || matmul_5_transpose_y) return false;

      auto matmul_1_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_1_in_2"));
      auto matmul_2_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_2_in_2"));
      auto matmul_3_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_3_in_2"));
      if (matmul_1_in_2.size() != 2 || matmul_2_in_2.size() != 2 ||
          matmul_3_in_2.size() != 2 ||
          matmul_1_in_2.at(0) != matmul_2_in_2.at(0) ||
          matmul_1_in_2.at(0) != matmul_3_in_2.at(0) ||
          matmul_1_in_2.at(1) != matmul_2_in_2.at(1) ||
          matmul_1_in_2.at(1) != matmul_3_in_2.at(1)) {
        return false;
      }

      auto add_1_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_1_in_2"));
      auto add_2_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_2_in_2"));
      auto add_3_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_3_in_2"));
      if (add_1_in_2.size() != 1 || add_2_in_2.size() != 1 ||
          add_3_in_2.size() != 1 || add_1_in_2.at(0) != add_2_in_2.at(0) ||
          add_1_in_2.at(0) != add_3_in_2.at(0)) {
        return false;
      }

      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();

    // W reshape.
    const auto &reshape_w_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto matmul_1_in_2 =
              pir::GetShapeFromValue(match_ctx.Tensor("matmul_1_in_2"));
          return {matmul_1_in_2.at(0), 1, matmul_1_in_2.at(1)};
        });
    const auto &reshape_5 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_5({&res.Tensor("matmul_1_in_2")},
              {&res.Tensor("reshape_5_out"), &res.NoneTensor()});
    const auto &reshape_6 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_6({&res.Tensor("matmul_2_in_2")},
              {&res.Tensor("reshape_6_out"), &res.NoneTensor()});
    const auto &reshape_7 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_7({&res.Tensor("matmul_3_in_2")},
              {&res.Tensor("reshape_7_out"), &res.NoneTensor()});

    // W combine.
    const auto &combine_1 = res.Op("builtin.combine");
    combine_1({&res.Tensor("reshape_5_out"),
               &res.Tensor("reshape_6_out"),
               &res.Tensor("reshape_7_out")},
              {&res.Tensor("combine_1_out")});

    const auto &concat_1 = res.Op("pd_op.concat", {{"axis", res.Int32Attr(1)}});
    res.Tensor("concat_1_out") = concat_1(res.Tensor("combine_1_out"));

    // Bias reshape.
    const auto &reshape_b_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto add_1_in_2 =
              pir::GetShapeFromValue(match_ctx.Tensor("add_1_in_2"));
          return {1, add_1_in_2.at(0)};
        });
    const auto &reshape_8 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_8({&res.Tensor("add_1_in_2")},
              {&res.Tensor("reshape_8_out"), &res.NoneTensor()});
    const auto &reshape_9 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_9({&res.Tensor("add_2_in_2")},
              {&res.Tensor("reshape_9_out"), &res.NoneTensor()});
    const auto &reshape_10 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_10({&res.Tensor("add_3_in_2")},
               {&res.Tensor("reshape_10_out"), &res.NoneTensor()});

    // Bias combine.
    const auto &combine_2 = res.Op("builtin.combine");
    combine_2({&res.Tensor("reshape_8_out"),
               &res.Tensor("reshape_9_out"),
               &res.Tensor("reshape_10_out")},
              {&res.Tensor("combine_2_out")});

    const auto &concat_2 = res.Op("pd_op.concat", {{"axis", res.Int32Attr(0)}});
    res.Tensor("concat_2_out") = concat_2(res.Tensor("combine_2_out"));

    const auto &head_number =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          const auto &full_int_array_1_value =
              match_ctx.Attr<std::vector<int64_t>>("full_int_array_1_value");
          return full_int_array_1_value.at(2);
        });
    const auto &alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("full_1_value");
        });
    const auto &multihead_matmul = res.Op("pd_op.multihead_matmul",
                                          {{"transpose_q", res.BoolAttr(false)},
                                           {"transpose_k", res.BoolAttr(true)},
                                           {"transpose_v", res.BoolAttr(false)},
                                           {"head_number", head_number},
                                           {"alpha", alpha}});
    multihead_matmul({&res.Tensor("matmul_1_in_1"),
                      &res.Tensor("concat_1_out"),
                      &res.Tensor("concat_2_out"),
                      &res.NoneTensor()},
                     {&res.Tensor("reshape_4_out")});
  }

  std::string name() const override {
    return "MultiHeadMatmulFuseNoBiasQKPattern";
  }
};

class MultiHeadMatmulFuseWithBiasQKPattern
    : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // The first path to matmul with scale (q).
    const auto &matmul_1 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_1_transpose_x")},
                {"transpose_y", src.Attr("matmul_1_transpose_y")}});
    src.Tensor("matmul_1_out") =
        matmul_1(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_1_in_2"));
    const auto &add_1 = src.Op("pd_op.add");
    src.Tensor("add_1_out") =
        add_1(src.Tensor("matmul_1_out"), src.Tensor("add_1_in_2"));
    const auto &full_int_array_1 =
        src.Op("pd_op.full_int_array",
               {{"value", src.Attr("full_int_array_1_value")}});
    const auto &reshape_1 = src.Op("pd_op.reshape");
    reshape_1({&src.Tensor("add_1_out"), &full_int_array_1()},
              {&src.Tensor("reshape_1_out"), &src.Tensor("reshape_1_xshape")});
    const auto &transpose_1 = src.Op("pd_op.transpose");
    src.Tensor("transpose_1_out") = transpose_1(src.Tensor("reshape_1_out"));
    const auto &full_1 =
        src.Op("pd_op.full", {{"value", src.Attr("full_1_value")}});
    const auto &scale = src.Op("pd_op.scale");
    src.Tensor("scale_out") = scale(src.Tensor("transpose_1_out"), full_1());

    // The second path to matmul (k).
    const auto &matmul_2 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_2_transpose_x")},
                {"transpose_y", src.Attr("matmul_2_transpose_y")}});
    src.Tensor("matmul_2_out") =
        matmul_2(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_2_in_2"));
    const auto &add_2 = src.Op("pd_op.add");
    src.Tensor("add_2_out") =
        add_2(src.Tensor("matmul_2_out"), src.Tensor("add_2_in_2"));
    const auto &full_int_array_2 = src.Op("pd_op.full_int_array");
    const auto &reshape_2 = src.Op("pd_op.reshape");
    reshape_2({&src.Tensor("add_2_out"), &full_int_array_2()},
              {&src.Tensor("reshape_2_out"), &src.Tensor("reshape_2_xshape")});
    const auto &transpose_2 = src.Op("pd_op.transpose");
    src.Tensor("transpose_2_out") = transpose_2(src.Tensor("reshape_2_out"));

    // The third path to matmul (v).
    const auto &matmul_3 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_3_transpose_x")},
                {"transpose_y", src.Attr("matmul_3_transpose_y")}});
    src.Tensor("matmul_3_out") =
        matmul_3(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_3_in_2"));
    const auto &add_3 = src.Op("pd_op.add");
    src.Tensor("add_3_out") =
        add_3(src.Tensor("matmul_3_out"), src.Tensor("add_3_in_2"));
    const auto &full_int_array_3 = src.Op("pd_op.full_int_array");
    const auto &reshape_3 = src.Op("pd_op.reshape");
    reshape_3({&src.Tensor("add_3_out"), &full_int_array_3()},
              {&src.Tensor("reshape_3_out"), &src.Tensor("reshape_3_xshape")});
    const auto &transpose_3 = src.Op("pd_op.transpose");
    src.Tensor("transpose_3_out") = transpose_3(src.Tensor("reshape_3_out"));

    // softmax(qk)v
    const auto &matmul_4 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_4_transpose_x")},
                {"transpose_y", src.Attr("matmul_4_transpose_y")}});
    src.Tensor("matmul_4_out") =
        matmul_4(src.Tensor("scale_out"), src.Tensor("transpose_2_out"));
    const auto &add_4 = src.Op("pd_op.add");
    src.Tensor("add_4_out") =
        add_4(src.Tensor("matmul_4_out"), src.Tensor("add_4_in_2"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("add_4_out"));
    const auto &matmul_5 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_5_transpose_x")},
                {"transpose_y", src.Attr("matmul_5_transpose_y")}});
    src.Tensor("matmul_5_out") =
        matmul_5(src.Tensor("softmax_out"), src.Tensor("transpose_3_out"));
    const auto &transpose_4 = src.Op("pd_op.transpose");
    src.Tensor("transpose_4_out") = transpose_4(src.Tensor("matmul_5_out"));
    const auto &full_int_array_4 = src.Op("pd_op.full_int_array");
    const auto &reshape_4 = src.Op("pd_op.reshape");
    reshape_4({&src.Tensor("transpose_4_out"), &full_int_array_4()},
              {&src.Tensor("reshape_4_out"), &src.Tensor("reshape_4_xshape")});

    //
    // Constraints.
    //
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;

      bool matmul_1_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_1_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_1_transpose_x || matmul_1_transpose_y) return false;

      bool matmul_2_transpose_x = match_ctx.Attr<bool>("matmul_2_transpose_x");
      bool matmul_2_transpose_y = match_ctx.Attr<bool>("matmul_2_transpose_y");
      if (matmul_2_transpose_x || matmul_2_transpose_y) return false;

      bool matmul_3_transpose_x = match_ctx.Attr<bool>("matmul_3_transpose_x");
      bool matmul_3_transpose_y = match_ctx.Attr<bool>("matmul_3_transpose_y");
      if (matmul_3_transpose_x || matmul_3_transpose_y) return false;

      bool matmul_4_transpose_x = match_ctx.Attr<bool>("matmul_4_transpose_x");
      bool matmul_4_transpose_y = match_ctx.Attr<bool>("matmul_4_transpose_y");
      if (matmul_4_transpose_x || !matmul_4_transpose_y) return false;

      bool matmul_5_transpose_x = match_ctx.Attr<bool>("matmul_5_transpose_x");
      bool matmul_5_transpose_y = match_ctx.Attr<bool>("matmul_5_transpose_y");
      if (matmul_5_transpose_x || matmul_5_transpose_y) return false;

      auto matmul_1_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_1_in_2"));
      auto matmul_2_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_2_in_2"));
      auto matmul_3_in_2 =
          pir::GetShapeFromValue(match_ctx.Tensor("matmul_3_in_2"));
      if (matmul_1_in_2.size() != 2 || matmul_2_in_2.size() != 2 ||
          matmul_3_in_2.size() != 2 ||
          matmul_1_in_2.at(0) != matmul_2_in_2.at(0) ||
          matmul_1_in_2.at(0) != matmul_3_in_2.at(0) ||
          matmul_1_in_2.at(1) != matmul_2_in_2.at(1) ||
          matmul_1_in_2.at(1) != matmul_3_in_2.at(1)) {
        return false;
      }

      auto add_1_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_1_in_2"));
      auto add_2_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_2_in_2"));
      auto add_3_in_2 = pir::GetShapeFromValue(match_ctx.Tensor("add_3_in_2"));
      if (add_1_in_2.size() != 1 || add_2_in_2.size() != 1 ||
          add_3_in_2.size() != 1 || add_1_in_2.at(0) != add_2_in_2.at(0) ||
          add_1_in_2.at(0) != add_3_in_2.at(0)) {
        return false;
      }

      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();

    // W reshape.
    const auto &reshape_w_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto matmul_1_in_2 =
              pir::GetShapeFromValue(match_ctx.Tensor("matmul_1_in_2"));
          return {matmul_1_in_2.at(0), 1, matmul_1_in_2.at(1)};
        });
    const auto &reshape_5 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_5({&res.Tensor("matmul_1_in_2")},
              {&res.Tensor("reshape_5_out"), &res.NoneTensor()});
    const auto &reshape_6 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_6({&res.Tensor("matmul_2_in_2")},
              {&res.Tensor("reshape_6_out"), &res.NoneTensor()});
    const auto &reshape_7 =
        res.Op("pd_op.reshape", {{"shape", reshape_w_shape_attr}});
    reshape_7({&res.Tensor("matmul_3_in_2")},
              {&res.Tensor("reshape_7_out"), &res.NoneTensor()});

    // W combine.
    const auto &combine_1 = res.Op("builtin.combine");
    combine_1({&res.Tensor("reshape_5_out"),
               &res.Tensor("reshape_6_out"),
               &res.Tensor("reshape_7_out")},
              {&res.Tensor("combine_1_out")});

    const auto &concat_1 = res.Op("pd_op.concat", {{"axis", res.Int32Attr(1)}});
    res.Tensor("concat_1_out") = concat_1(res.Tensor("combine_1_out"));

    // Bias reshape.
    const auto &reshape_b_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto add_1_in_2 =
              pir::GetShapeFromValue(match_ctx.Tensor("add_1_in_2"));
          return {1, add_1_in_2.at(0)};
        });
    const auto &reshape_8 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_8({&res.Tensor("add_1_in_2")},
              {&res.Tensor("reshape_8_out"), &res.NoneTensor()});
    const auto &reshape_9 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_9({&res.Tensor("add_2_in_2")},
              {&res.Tensor("reshape_9_out"), &res.NoneTensor()});
    const auto &reshape_10 =
        res.Op("pd_op.reshape", {{"shape", reshape_b_shape_attr}});
    reshape_10({&res.Tensor("add_3_in_2")},
               {&res.Tensor("reshape_10_out"), &res.NoneTensor()});

    // Bias combine.
    const auto &combine_2 = res.Op("builtin.combine");
    combine_2({&res.Tensor("reshape_8_out"),
               &res.Tensor("reshape_9_out"),
               &res.Tensor("reshape_10_out")},
              {&res.Tensor("combine_2_out")});

    const auto &concat_2 = res.Op("pd_op.concat", {{"axis", res.Int32Attr(0)}});
    res.Tensor("concat_2_out") = concat_2(res.Tensor("combine_2_out"));

    const auto &head_number =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          const auto &full_int_array_1_value =
              match_ctx.Attr<std::vector<int64_t>>("full_int_array_1_value");
          return full_int_array_1_value.at(2);
        });
    const auto &alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("full_1_value");
        });
    const auto &multihead_matmul = res.Op("pd_op.multihead_matmul",
                                          {{"transpose_q", res.BoolAttr(false)},
                                           {"transpose_k", res.BoolAttr(true)},
                                           {"transpose_v", res.BoolAttr(false)},
                                           {"head_number", head_number},
                                           {"alpha", alpha}});
    multihead_matmul({&res.Tensor("matmul_1_in_1"),
                      &res.Tensor("concat_1_out"),
                      &res.Tensor("concat_2_out"),
                      &res.Tensor("add_4_in_2")},
                     {&res.Tensor("reshape_4_out")});
  }

  std::string name() const override {
    return "MultiHeadMatmulFuseWithBiasQKPattern";
  }
};

class MultiHeadMatmulFusePass : public pir::PatternRewritePass {
 public:
  MultiHeadMatmulFusePass()
      : pir::PatternRewritePass("multihead_matmul_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(MultiHeadMatmulFuseNoBiasQKPattern().Build(context));
    ps.Add(MultiHeadMatmulFuseWithBiasQKPattern().Build(context));
    // Add other attention variant fuse pattern.

    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateMultiHeadMatmulFusePass() {
  return std::make_unique<MultiHeadMatmulFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(multihead_matmul_fuse_pass, MultiHeadMatmulFusePass);
