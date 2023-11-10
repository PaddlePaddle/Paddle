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

#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class SumOpPattern : public pir::drr::DrrPatternBase<SumOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &sum = pattern.Op(paddle::dialect::SumOp::name(),
                                 {{"dtype", pattern.Attr("dtype")},
                                  {"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = sum(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_sum =
        res.Op(cinn::dialect::ReduceSumOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_sum(res.Tensor("arg0"));
  }
};

class MaxOpPattern : public pir::drr::DrrPatternBase<MaxOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::MaxOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceMaxOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }
};

// class ReshapeOpPattern : public pir::drr::DrrPatternBase<ReshapeOpPattern> {
//  public:
//   void operator()(pir::drr::DrrPatternContext *ctx) const override {
//     // Source Pattern
//     pir::drr::SourcePattern pattern = ctx->SourcePattern();
//     const auto &full_int_array =
//         pattern.Op(paddle::dialect::FullIntArrayOp::name(),
//                    {{"value", pattern.Attr("axis_info")},
//                     {"dtype", pattern.Attr("dtype_2")},
//                     {"place", pattern.Attr("place_2")}});

//     const auto &pd_reshape = pattern.Op(paddle::dialect::ReshapeOp::name(),
//                                     {});
//     pattern.Tensor("ret") = pd_reshape(pattern.Tensor("arg0"),
//     full_int_array());

//     // Result patterns
//     pir::drr::ResultPattern res = pattern.ResultPattern();
//     const auto &cinn_reshape =
//         res.Op(cinn::dialect::ReshapeCOp::name(),
//                {{"shape", pattern.Attr("axis_info")}});
//     res.Tensor("ret") = cinn_reshape(res.Tensor("arg0"));
//   }
// };

class ReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ReshapeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto scale_factor_gen_op =
        op->operand_source(1).dyn_cast<pir::OpResult>().owner();

    if (auto full_op =
            scale_factor_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>()) {
      // sacle is generator by full op
      // get attribute value from full op

      auto out_shape_attr =
          full_op.attribute("value").dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int> vec_out_shape;
      if (out_shape_attr.size() > 0) {
        PADDLE_ENFORCE_EQ(
            out_shape_attr[0].isa<::pir::Int64Attribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be ir::Int64Attribute"));
        for (size_t i = 0; i < out_shape_attr.size(); ++i) {
          vec_out_shape.push_back(
              out_shape_attr[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }

      auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeCOp>(
          op->operand_source(0).dyn_cast<pir::OpResult>(), vec_out_shape);
      rewriter.ReplaceAllUsesWith(op.result(0), cinn_reshape.result(0));
      rewriter.EraseOp(op);
      rewriter.EraseOp(full_op);

      return true;
    }
    return false;
  }
};

PdOpToCinnOpPass::PdOpToCinnOpPass() : pir::Pass("pd_to_cinn_pass", 1) {}

bool PdOpToCinnOpPass::Initialize(pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add(SumOpPattern().Build(context));
  ps.Add(MaxOpPattern().Build(context));
  ps.Add<ReshapeOpPattern>(context);

  patterns_ = ::pir::FrozenRewritePatternSet(std::move(ps));
  return true;
}

void PdOpToCinnOpPass::Run(pir::Operation *op) {
  pir::GreedyRewriteConfig cfg;
  cfg.use_top_down_traversal = true;
  cfg.max_iterations = 10;
  pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
}

bool PdOpToCinnOpPass::CanApplyOn(pir::Operation *op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

void PdOp2CinnOpConverter(::pir::Program *program) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<PdOpToCinnOpPass>());

  pm.Run(program);
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
