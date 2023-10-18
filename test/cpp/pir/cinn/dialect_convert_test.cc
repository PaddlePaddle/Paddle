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

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/pir/transforms/dead_code_elimination_pass.h"

class PDSum2CINNReduceSumPattern
    : public pir::drr::DrrPatternBase<PDSum2CINNReduceSumPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full_int_array = pat.Op("pd_op.full_int_array",
                                        {{"value", pat.Attr("axis_info")},
                                         {"dtype", pat.Attr("dtype_2")},
                                         {"place", pat.Attr("place_2")}});

    const auto &sum = pat.Op(
        "pd_op.sum",
        {{"dtype", pat.Attr("dtype")}, {"keepdim", pat.Attr("keep_dim")}});
    pat.Tensor("ret") = sum(pat.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &cinn_reduce_sum = res.Op(
        "cinn_op.reduce_sum",
        {{"axis", pat.Attr("axis_info")}, {"keep_dim", pat.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_sum(res.Tensor("arg0"));
  }
};

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  auto sum_op =
      builder.Build<paddle::dialect::SumOp>(full_input_op.result(0),
                                            std::vector<int64_t>({-1}),
                                            phi::DataType::FLOAT32,
                                            true);
  auto relu_op = builder.Build<paddle::dialect::ReluOp>(sum_op.result(0));
  auto exp_op = builder.Build<paddle::dialect::ExpOp>(sum_op.result(0));
}

class DrrPatternRewritePass : public pir::Pass {
 public:
  DrrPatternRewritePass() : pir::Pass("DrrPatternRewritePass", 1) {}

  bool Initialize(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(PDSum2CINNReduceSumPattern().Build(context));

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
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  program.Print(std::cout);

  //   EXPECT_EQ(program.block()->size(), 14u);

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<DrrPatternRewritePass>());
  // pm.EnablePassTiming();
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);

  program.Print(std::cout);
  //   EXPECT_EQ(program.block()->size(), 7u);
}
