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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/pir/transforms/dead_code_elimination_pass.h"

class RemoveRedundentReshapePattern
    : public pir::drr::DrrPatternBase<RemoveRedundentReshapePattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source patterns
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &reshape1 = pat.Op("pd_op.reshape");
    const auto &reshape2 = pat.Op("pd_op.reshape");

    reshape1({&pat.Tensor("arg0"), &pat.Tensor("shape0")},
             {&pat.Tensor("out1"), &pat.Tensor("xshape_0")});
    reshape2({&pat.Tensor("out1"), &pat.Tensor("shape1")},
             {&pat.Tensor("ret"), &pat.Tensor("xshape_1")});

    // Result patterns
    pir::drr::ResultPattern res = pat.ResultPattern();
    res.Op("pd_op.reshape")({&res.Tensor("arg0"), &res.Tensor("shape1")},
                            {&res.Tensor("ret"), &res.Tensor("xshape_1")});
  }
};

class FoldExpandToConstantPattern
    : public pir::drr::DrrPatternBase<FoldExpandToConstantPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full1 = pat.Op("pd_op.full",
                               {{"shape", pat.Attr("shape_1")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    const auto &full_int_array1 =
        pat.Op("pd_op.full_int_array",
               {{"value", pat.Attr("expand_shape_value")},
                {"dtype", pat.Attr("dtype_2")},
                {"place", pat.Attr("place_2")}});
    const auto &expand = pat.Op("pd_op.expand");
    pat.Tensor("ret") = expand(full1(), full_int_array1());

    // Result patterns
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &full2 = res.Op("pd_op.full",
                               {{"shape", pat.Attr("expand_shape_value")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    res.Tensor("ret") = full2();
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

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullIntArrayOp full_int_array_op =
      builder.Build<paddle::dialect::FullIntArrayOp>(
          std::vector<int64_t>{4, 3, 16, 16},
          phi::DataType::FLOAT32,
          phi::CPUPlace());

  paddle::dialect::ExpandOp expand_op =
      builder.Build<paddle::dialect::ExpandOp>(full_input_op.out(),
                                               full_int_array_op.out());

  paddle::dialect::ReshapeOp reshape_op1 =
      builder.Build<paddle::dialect::ReshapeOp>(
          expand_op.out(), std::vector<int64_t>{16, 3, 4, 16});

  paddle::dialect::ReshapeOp reshape_op2 =
      builder.Build<paddle::dialect::ReshapeOp>(
          reshape_op1.out(), std::vector<int64_t>{16, 3, 4, 16});

  paddle::dialect::ReluOp relu_op =
      builder.Build<paddle::dialect::ReluOp>(reshape_op2.out());

  paddle::dialect::CastOp cast_op1 = builder.Build<paddle::dialect::CastOp>(
      relu_op.out(), phi::DataType::FLOAT64);

  paddle::dialect::CastOp cast_op2 = builder.Build<paddle::dialect::CastOp>(
      cast_op1.out(), phi::DataType::FLOAT32);

  paddle::dialect::TransposeOp transpose_op1 =
      builder.Build<paddle::dialect::TransposeOp>(cast_op2.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  paddle::dialect::TransposeOp transpose_op2 =
      builder.Build<paddle::dialect::TransposeOp>(transpose_op1.out(),
                                                  std::vector<int>{1, 0, 2, 3});

  paddle::dialect::ReluOp relu_op_second =
      builder.Build<paddle::dialect::ReluOp>(transpose_op2.out());

  builder.Build<paddle::dialect::FetchOp>(relu_op_second.out(), "out", 0);
}

class DrrPatternRewritePass : public pir::Pass {
 public:
  DrrPatternRewritePass() : pir::Pass("DrrPatternRewritePass", 1) {}

  bool Initialize(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(RemoveRedundentReshapePattern().Build(context));
    ps.Add(RemoveRedundentTransposePattern().Build(context));
    ps.Add(RemoveRedundentCastPattern().Build(context));
    ps.Add(RemoveUselessCastPattern().Build(context));
    ps.Add(FoldExpandToConstantPattern().Build(context));

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation *op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(pir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

TEST(DrrTest, drr_demo) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 14u);

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<DrrPatternRewritePass>());
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  // pm.EnablePassTiming();
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
  EXPECT_EQ(program.block()->size(), 7u);
}
