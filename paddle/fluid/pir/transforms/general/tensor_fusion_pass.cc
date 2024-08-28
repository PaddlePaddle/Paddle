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

#include "paddle/fluid/pir/transforms/general/tensor_fusion_pass.h"

#include <glog/logging.h>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class TensorFusionPass : public pir::PatternRewritePass {
 public:
  TensorFusionPass() : pir::PatternRewritePass("tensor_fusion_pass", 2) {}

  class CastFusionPattern : public paddle::drr::DrrPatternBase {
   public:
    std::string name() const override { return "CastFusionPattern"; }

    void operator()(paddle::drr::DrrPatternContext *ctx) const override {
      paddle::drr::SourcePattern src_pat = ctx->SourcePattern();
      const auto &cast_op1 = src_pat.Op(paddle::dialect::CastOp::name(),
                                        {{"dtype", src_pat.Attr("dtype1")}});
      const auto &cast_op2 = src_pat.Op(paddle::dialect::CastOp::name(),
                                        {{"dtype", src_pat.Attr("dtype2")}});
      src_pat.Tensor("out1") = cast_op1(src_pat.Tensor("x1"));
      src_pat.Tensor("out2") = cast_op2(src_pat.Tensor("x2"));

      src_pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        std::vector<int64_t> x1_shape =
            pir::GetShapeFromValue(match_ctx.Tensor("x1"));
        std::vector<int64_t> x2_shape =
            pir::GetShapeFromValue(match_ctx.Tensor("x2"));
        bool equal_shape = x1_shape.size() == x2_shape.size();
        if (equal_shape) {
          for (size_t i = 0; i < x1_shape.size(); ++i) {
            if (x1_shape[i] != x2_shape[i]) {
              equal_shape = false;
              break;
            }
          }
        }
        VLOG(4) << "equal_shape: " << equal_shape;
        const auto &cast1_type = match_ctx.Attr<phi::DataType>("dtype1");
        const auto &cast2_type = match_ctx.Attr<phi::DataType>("dtype2");
        bool equal_dtype = cast1_type == cast2_type;
        VLOG(4) << "equal_dtype: " << equal_dtype;
        bool rtn = equal_shape && equal_dtype;
        VLOG(4) << "rtn: " << rtn;
        return rtn;
      });

      paddle::drr::ResultPattern res_pat = src_pat.ResultPattern();
      const auto &buitin_combine_op = res_pat.Op(pir::CombineOp::name());
      const auto &concat_and_relocate_op =
          res_pat.Op(paddle::dialect::ConcatAndRelocate_Op::name());
      const auto &cast_op = res_pat.Op(paddle::dialect::CastOp::name(),
                                       {{"dtype", src_pat.Attr("dtype1")}});
      const auto &split_op = res_pat.Op(paddle::dialect::SplitOp::name(),
                                        {{"axis", res_pat.Int32Attr(-1)}});
      const auto &buitin_split_op = res_pat.Op(pir::SplitOp::name());

      buitin_combine_op({&res_pat.Tensor("x1"), &res_pat.Tensor("x2")},
                        {&res_pat.Tensor("x")});
      res_pat.Tensor("fused_x") = concat_and_relocate_op(res_pat.Tensor("x"));
      res_pat.Tensor("fuesd_out") = cast_op(res_pat.Tensor("fused_x"));
      res_pat.Tensor("split_outs") = split_op(res_pat.Tensor("fuesd_out"));
      buitin_split_op({&res_pat.Tensor("split_outs")},
                      {&res_pat.Tensor("out1"), &res_pat.Tensor("out2")});
    }
  };

  // class MatmulFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "MatmulFusionPattern"; }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class SwigluFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "SwigluFusionPattern"; }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class SliceFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "SliceFusionPattern"; }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class ReduceScatterFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "ReduceScatterFusionPattern";
  //   }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class Adamw_FusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "Adamw_FusionPattern"; }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class AllGatherFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override { return "AllGatherFusionPattern"; }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  // class CoalesceWithSplitFusionPattern : public paddle::drr::DrrPatternBase {
  //  public:
  //   std::string name() const override {
  //     return "CoalesceWithSplitFusionPattern";
  //   }

  //   void operator()(paddle::drr::DrrPatternContext *ctx) const override {}
  // };

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<CastFusionPattern>(context));
    // ps.Add(paddle::drr::Create<MatmulFusionPattern>(context));
    // ps.Add(paddle::drr::Create<SwigluFusionPattern>(context));
    // ps.Add(paddle::drr::Create<SliceFusionPattern>(context));
    // ps.Add(paddle::drr::Create<ReduceScatterFusionPattern>(context));
    // ps.Add(paddle::drr::Create<Adamw_FusionPattern>(context));
    // ps.Add(paddle::drr::Create<AllGatherFusionPattern>(context));
    // // NOTE(zhangbo): This Pattern is used to eliminate the redundant ops of
    // // coalesce_tensor and split generated after the above Pattern hits. It
    // // needs to be registered to the end.
    // ps.Add(paddle::drr::Create<CoalesceWithSplitFusionPattern>(context));
    return ps;
  }

  // void PatternRewritePass::Run(Operation* op) {
  //   auto [_, num_rewrites] =
  //       ApplyPatternsGreedily(op, patterns_, InitializeConfig());
  //   AddStatistics(num_rewrites);
  // }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTensorFusionPass() {
  return std::make_unique<TensorFusionPass>();
}

}  // namespace pir

REGISTER_IR_PASS(tensor_fusion_pass, TensorFusionPass);
