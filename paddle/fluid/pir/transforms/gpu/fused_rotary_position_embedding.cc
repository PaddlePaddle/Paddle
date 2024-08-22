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
 private:
  bool with_transpose_;

 public:
  explicit FusedRotaryPositionEmbeddingPattern(bool with_transpose)
      : with_transpose_(with_transpose) {}

  std::string name() const override { return "RotaryPositionEmbeddingPattern"; }

  uint32_t benefit() const override { return 2; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &squeeze = pat.Op(paddle::dialect::SqueezeOp::name());
    const auto &squeeze_1 = pat.Op(paddle::dialect::SqueezeOp::name());

    const auto &gather_nd = pat.Op(paddle::dialect::GatherNdOp::name());
    const auto &gather_nd_1 = pat.Op(paddle::dialect::GatherNdOp::name());
    const auto &unsqueeze = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_1 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_2 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_4 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_1 = pat.Op(paddle::dialect::AddOp::name());
    const auto &multiply1 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply2 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply3 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply4 = pat.Op(paddle::dialect::MultiplyOp::name());

    const auto &slice_q = pat.Op(paddle::dialect::SliceOp::name());
    const auto &slice_q_1 = pat.Op(paddle::dialect::SliceOp::name());

    const auto &slice_k = pat.Op(paddle::dialect::SliceOp::name());

    const auto &slice_k_1 = pat.Op(paddle::dialect::SliceOp::name());

    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &full_op_1 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"value", pat.Attr("full_op_1")}});
    const auto &full_op_2 = pat.Op(paddle::dialect::FullOp::name());
    const auto &full_op_3 = pat.Op(paddle::dialect::FullOp::name());

    const auto &scale_op = pat.Op(paddle::dialect::ScaleOp::name());

    const auto &scale_op_k = pat.Op(paddle::dialect::ScaleOp::name());

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_2 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_3 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_4 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_5 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_6 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_7 = pat.Op(paddle::dialect::FullIntArrayOp::name());
    const auto &full_8 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_8_value")}});
    const auto &full_9 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_9_value")}});
    const auto &full_10 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_10_value")}});
    const auto &full_11 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_11_value")}});
    const auto &full_12 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_12_value")}});
    const auto &full_13 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_13_value")}});
    const auto &full_14 = pat.Op(paddle::dialect::FullIntArrayOp::name());

    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine = pat.Op(pir::CombineOp::name());
    const auto &concat_op_k = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine_k = pat.Op(pir::CombineOp::name());

    squeeze({&pat.Tensor("cos"), &full_13()}, {&pat.Tensor("squeeze_out_cos")});

    squeeze_1({&pat.Tensor("sin"), &full_12()},
              {&pat.Tensor("squeeze_out_sin")});

    unsqueeze({&pat.Tensor("position_ids"), &full_11()},
              {&pat.Tensor("unsqueeze_s_out_cos")});

    if (with_transpose_) {
      const auto &transpose_1 = pat.Op(paddle::dialect::TransposeOp::name(),
                                       {{"perm", pat.Attr("perm_1")}});
      pat.Tensor("transpose_1_cos") =
          transpose_1(pat.Tensor("squeeze_out_cos"));

      const auto &transpose_2 = pat.Op(paddle::dialect::TransposeOp::name(),
                                       {{"perm", pat.Attr("perm_2")}});
      pat.Tensor("transpose_2_sin") =
          transpose_2(pat.Tensor("squeeze_out_sin"));

      pat.Tensor("gather_nd_out_cos") = gather_nd(
          pat.Tensor("transpose_1_cos"), pat.Tensor("unsqueeze_s_out_cos"));

      pat.Tensor("gather_nd_out_sin") = gather_nd_1(
          pat.Tensor("transpose_2_sin"), pat.Tensor("unsqueeze_s_out_sin"));
    } else {
      pat.Tensor("gather_nd_out_cos") = gather_nd(
          pat.Tensor("squeeze_out_cos"), pat.Tensor("unsqueeze_s_out_cos"));
      pat.Tensor("gather_nd_out_sin") = gather_nd_1(
          pat.Tensor("squeeze_out_sin"), pat.Tensor("unsqueeze_s_out_sin"));
    }

    unsqueeze_1({&pat.Tensor("gather_nd_out_cos"), &full_10()},
                {&pat.Tensor("unsqueeze_out_cos")});

    unsqueeze_4({&pat.Tensor("position_ids"), &full_8()},
                {&pat.Tensor("unsqueeze_s_out_sin")});

    unsqueeze_2({&pat.Tensor("gather_nd_out_sin"), &full_9()},
                {&pat.Tensor("unsqueeze_out_sin")});

    pat.Tensor("tmp_25") =
        multiply1(pat.Tensor("q"), pat.Tensor("unsqueeze_out_cos"));

    pat.Tensor("q_slice_out1") = slice_q(pat.Tensor("q"), full_1(), full_2());

    pat.Tensor("q_slice_out2") = slice_q_1(pat.Tensor("q"), full_3(), full_4());

    scale_op({&pat.Tensor("q_slice_out2"), &full_op()},
             {&pat.Tensor("scale_out")});

    std::vector<const paddle::drr::Tensor *> combine_in;
    combine_in.push_back(&pat.Tensor("scale_out"));
    combine_in.push_back(&pat.Tensor("q_slice_out1"));
    combine(combine_in, {&pat.Tensor("combine_out")});

    concat_op({&pat.Tensor("combine_out"), &full_op_3()},
              {&pat.Tensor("concat_out")});

    pat.Tensor("tmp_27") =
        multiply3(pat.Tensor("concat_out"), pat.Tensor("unsqueeze_out_sin"));

    pat.Tensor("out_q") = add(pat.Tensor("tmp_25"), pat.Tensor("tmp_27"));

    pat.Tensor("tmp_29") =
        multiply2(pat.Tensor("k"), pat.Tensor("unsqueeze_out_cos"));

    pat.Tensor("k_slice_out1") = slice_k(pat.Tensor("k"), full_5(), full_6());

    pat.Tensor("k_slice_out2") =
        slice_k_1(pat.Tensor("k"), full_7(), full_14());

    scale_op_k({&pat.Tensor("k_slice_out2"), &full_op_1()},
               {&pat.Tensor("scale_out_k")});

    std::vector<const paddle::drr::Tensor *> combine_in_k;
    combine_in_k.push_back(&pat.Tensor("scale_out_k"));
    combine_in_k.push_back(&pat.Tensor("k_slice_out1"));
    combine_k(combine_in_k, {&pat.Tensor("combine_out_k")});

    concat_op_k({&pat.Tensor("combine_out_k"), &full_op_2()},
                {&pat.Tensor("concat_out_k")});

    pat.Tensor("tmp_31") =
        multiply4(pat.Tensor("concat_out_k"), pat.Tensor("unsqueeze_out_sin"));

    pat.Tensor("out_k") = add_1(pat.Tensor("tmp_29"), pat.Tensor("tmp_31"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
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

      bool check_passes = true;
      auto axis = match_ctx.Attr<std::vector<int64_t>>("full_13_value");
      auto axis_2 = match_ctx.Attr<std::vector<int64_t>>("full_12_value");
      check_passes = check_passes && check_axes(axis) && check_axes(axis_2);

      auto check_unsqueeze_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes_1 = {-1};
        std::vector<int64_t> expected_axes_2 = {2};
        if (axes.size() != 1) {
          return false;
        }
        if (axes[0] == expected_axes_1[0] || axes[0] == expected_axes_2[0]) {
          return true;
        }
        return false;
      };

      auto unsqueeze_axis =
          match_ctx.Attr<std::vector<int64_t>>("full_11_value");
      auto unsqueeze_axis_1 =
          match_ctx.Attr<std::vector<int64_t>>("full_10_value");
      auto unsqueeze_axis_2 =
          match_ctx.Attr<std::vector<int64_t>>("full_8_value");
      auto unsqueeze_axis_3 =
          match_ctx.Attr<std::vector<int64_t>>("full_9_value");

      check_passes = check_passes && check_unsqueeze_axes(unsqueeze_axis) &&
                     check_unsqueeze_axes(unsqueeze_axis_1) &&
                     check_unsqueeze_axes(unsqueeze_axis_2) &&
                     check_unsqueeze_axes(unsqueeze_axis_3);

      return check_passes;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_rotary_position_embedding = res.Op(
        paddle::dialect::FusedRotaryPositionEmbeddingOp::name(),
        {
            // TODO(lizexu123): This pass only supports the case where
            // use_neox_rotary_style is false. When use_neox_rotary_style is
            // true, the source pattern needs to be modified accordingly.
            {"use_neox_rotary_style", res.BoolAttr(false)},
            {"time_major", res.BoolAttr(false)},
            {"rotary_emb_base", res.Float32Attr(10000.0)},
        });

    fused_rotary_position_embedding(
        {&res.Tensor("q"),
         &res.Tensor("k"),
         &res.InputNoneTensor(),
         &res.Tensor("sin"),
         &res.Tensor("cos"),
         &res.Tensor("position_ids")},
        {&res.Tensor("out_q"), &res.Tensor("out_k"), &res.OutputNoneTensor()});
  }
};
class FusedRotaryPositionEmbeddingPass : public pir::PatternRewritePass {
 public:
  FusedRotaryPositionEmbeddingPass()
      : pir::PatternRewritePass("fused_rotary_position_embedding_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedRotaryPositionEmbeddingPattern>(context,
                                                                    true));
    ps.Add(paddle::drr::Create<FusedRotaryPositionEmbeddingPattern>(context,
                                                                    false));
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
