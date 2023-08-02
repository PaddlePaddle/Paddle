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

#include <gtest/gtest.h>

#include <glog/logging.h>

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"
#include "paddle/ir/pattern_rewrite/drr/drr_rewrite_pattern.h"
#include "paddle/ir/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/fluid/ir/dialect/pd_op.h"

struct RemoveRedundentReshapeFunctor {
  void operator()(ir::drr::DrrPatternContext *ctx) {
    // Source patterns：待匹配的子图
    ir::drr::SourcePattern pat = ctx->SourcePattern();
    VLOG(1) << "###### reshape";
    const auto &reshape = pat.Op("reshape");
    VLOG(1) << "###### reshape reshape";
    pat.Tensor("ret") = reshape(reshape(pat.Tensor("arg0")));

    // Result patterns：要替换为的子图
    ir::drr::ResultPattern res = pat.ResultPattern();
    VLOG(1) << "###### RES";
    res.Tensor("ret") = res.Op("reshape")(res.Tensor("arg0"));
  }
};

class RemoveRedundentReshapePattern
    : public ir::drr::DrrRewritePattern<paddle::dialect::ReshapeOp,
                                        RemoveRedundentReshapeFunctor> {
 public:
  using ir::drr::DrrRewritePattern<
      paddle::dialect::ReshapeOp,
      RemoveRedundentReshapeFunctor>::DrrRewritePattern;
};

void BuildProgram(ir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::ReshapeOp reshape_op_first =
      builder.Build<paddle::dialect::ReshapeOp>(
          full_input_op.out(), std::vector<int64_t>{16, 3, 4, 16});

  paddle::dialect::ReshapeOp reshape_op_second =
      builder.Build<paddle::dialect::ReshapeOp>(
          reshape_op_first.out(), std::vector<int64_t>{16, 3, 4, 16});

  paddle::dialect::ReluOp relu_op =
      builder.Build<paddle::dialect::ReluOp>(reshape_op_second.out());

  builder.Build<paddle::dialect::FetchOp>(relu_op.out(), "out", 0);
}

std::unique_ptr<RemoveRedundentReshapePattern> CreateDrrPatternRewritePass(
    ir::IrContext *ir_ctx) {
  return std::make_unique<RemoveRedundentReshapePattern>(ir_ctx, 1);
}

class TestPass : public ir::Pass {
 public:
  TestPass() : ir::Pass("TestPass", 1) {}

  bool Initialize(ir::IrContext *context) override {
    ir::RewritePatternSet ps(context);
    ps.Add(std::move(CreateDrrPatternRewritePass(context)));

    patterns_ = ir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(ir::Operation *op) override {
    ir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    ir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }

 private:
  ir::FrozenRewritePatternSet patterns_;
};

TEST(DrrTest, drr) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 7u);

  ir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<TestPass>());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
}
