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

#include "paddle/fluid/pir/transforms/fusion/transpose_flatten_concat_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class NTransposeFlattenConcatFusePattern : public paddle::drr::DrrPatternBase {
 public:
  explicit NTransposeFlattenConcatFusePattern(int transpose_flatten_time) {
    transpose_flatten_time_ = transpose_flatten_time;
  }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::vector<const paddle::drr::Tensor *> combine_in;
    for (int i = 0; i < transpose_flatten_time_; i++) {
      const auto &transpose_op =
          pat.Op(paddle::dialect::TransposeOp::name(),
                 {{"perm", pat.Attr("perm_" + std::to_string(i))}});
      const auto &flatten_op =
          pat.Op(paddle::dialect::FlattenOp::name(),
                 {{"start_axis", pat.Attr("start_axis_" + std::to_string(i))},
                  {"stop_axis", pat.Attr("stop_axis_" + std::to_string(i))}});
      transpose_op({&pat.Tensor("transpose_in_" + std::to_string(i))},
                   {&pat.Tensor("transpose_out_" + std::to_string(i))});
      flatten_op({&pat.Tensor("transpose_out_" + std::to_string(i))},
                 {&pat.Tensor("flatten_out_" + std::to_string(i)),
                  &pat.Tensor("flatten_xshape_" + std::to_string(i))});
      combine_in.push_back(&pat.Tensor("flatten_out_" + std::to_string(i)));
    }
    const auto &combine_op = pat.Op(pir::CombineOp::name());
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"value", pat.Attr("full_value")}});
    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    combine_op(combine_in, {&pat.Tensor("combine_out")});
    concat_op({&pat.Tensor("combine_out"), &full_op()},
              {&pat.Tensor("concat_out")});
    pat.RequireNativeCall(
        [this](const paddle::drr::MatchContext &match_ctx) -> bool {
          auto flatten_out_shape_0 =
              pir::GetShapeFromValue(match_ctx.Tensor("flatten_out_0"));
          if (flatten_out_shape_0.size() != 2) {
            return false;
          }
          if (transpose_flatten_time_ < 2) {
            return true;
          }

          std::vector<int32_t> perm_0 =
              match_ctx.Attr<std::vector<int32_t>>("perm_0");
          int flatten_start_0 = match_ctx.Attr<int>("start_axis_0");
          int flatten_stop_0 = match_ctx.Attr<int>("stop_axis_0");
          for (int i = 1; i < transpose_flatten_time_; i++) {
            auto flatten_out_shape = pir::GetShapeFromValue(
                match_ctx.Tensor("flatten_out_" + std::to_string(i)));
            if (flatten_out_shape.size() != 2) {
              return false;
            }
            auto tmp_perm = match_ctx.Attr<std::vector<int32_t>>(
                "perm_" + std::to_string(i));
            auto tmp_flatten_start =
                match_ctx.Attr<int>("start_axis_" + std::to_string(i));
            auto tmp_flatten_stop =
                match_ctx.Attr<int>("stop_axis_" + std::to_string(i));
            if (perm_0.size() != tmp_perm.size()) {
              return false;
            }
            for (size_t j = 0; j < perm_0.size(); j++) {
              if (perm_0[j] != tmp_perm[j]) {
                return false;
              }
            }
            if (flatten_start_0 != tmp_flatten_start ||
                flatten_stop_0 != tmp_flatten_stop) {
              return false;
            }
          }
          return true;
        });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &res_trans_axis = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          return match_ctx.Attr<std::vector<int>>("perm_0");
        });
    const auto &res_flatten_axis =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          int start_axis = match_ctx.Attr<int>("start_axis_0");
          int stop_axis = match_ctx.Attr<int>("stop_axis_0");
          if (start_axis == stop_axis) {
            return start_axis;
          } else if (start_axis == 0) {
            return stop_axis + 1;
          } else {
            return start_axis;
          }
        });
    const auto &res_concat_axis =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          return static_cast<int>(match_ctx.Attr<float>("full_value"));
        });
    const auto &fusion_transpose_flatten_concat_op =
        res.Op(paddle::dialect::FusionTransposeFlattenConcatOp::name(),
               {
                   {"trans_axis", res_trans_axis},
                   {"flatten_axis", res_flatten_axis},
                   {"concat_axis", res_concat_axis},
               });
    std::vector<const paddle::drr::Tensor *> x_in;
    for (int i = 0; i < transpose_flatten_time_; i++) {
      x_in.push_back(&res.Tensor("transpose_in_" + std::to_string(i)));
    }
    const auto &combine_2 = res.Op(pir::CombineOp::name());
    combine_2(x_in, {&res.Tensor("combine_2_out")});
    fusion_transpose_flatten_concat_op({&res.Tensor("combine_2_out")},
                                       {&res.Tensor("concat_out")});
  }

  std::string name() const override {
    return "NTransposeFlattenConcatFusePattern_" +
           std::to_string(transpose_flatten_time_);
  }

 private:
  int transpose_flatten_time_;
};

class TransposeFlattenConcatFusePass : public pir::PatternRewritePass {
 public:
  TransposeFlattenConcatFusePass()
      : pir::PatternRewritePass("transpose_flatten_concat_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    for (int pattern_num = 1; pattern_num <= 6; ++pattern_num) {
      ps.Add(NTransposeFlattenConcatFusePattern(pattern_num).Build(context));
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTransposeFlattenConcatFusePass() {
  return std::make_unique<TransposeFlattenConcatFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(transpose_flatten_concat_fuse_pass,
                 TransposeFlattenConcatFusePass);
