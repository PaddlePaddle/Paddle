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
#include <cstdint>
#include <memory>
#include <vector>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/drr/api/drr_pattern_base.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/ir/pattern_rewrite/pattern_rewrite_driver.h"

class MultiHeadMatmulFusePattern
    : public ir::drr::DrrPatternBase<MultiHeadMatmulFusePattern> {
 public:
  void operator()(ir::drr::DrrPatternContext *ctx) const override {
    //
    // Source Pattern.
    //
    ir::drr::SourcePattern src = ctx->SourcePattern();
    // The first path to matmul with scale (q).
    const auto &matmul_1 =
        src.Op("pd.matmul",
               {{"transpose_x", src.Attr("matmul_1_transpose_x")},
                {"transpose_y", src.Attr("matmul_1_transpose_y")}});
    src.Tensor("matmul_1_out") =
        matmul_1(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_1_in_2"));
    const auto &add_1 = src.Op("pd.add");
    src.Tensor("add_1_out") =
        add_1(src.Tensor("matmul_1_out"), src.Tensor("add_1_in_2"));
    const auto &full_int_array_1 = src.Op(
        "pd.full_int_array", {{"value", src.Attr("full_int_array_1_value")}});
    const auto &reshape_1 = src.Op("pd.reshape");
    reshape_1({&src.Tensor("add_1_out"), &full_int_array_1()},
              {&src.Tensor("reshape_1_out"), &src.Tensor("reshape_1_xshape")});
    const auto &transpose_1 = src.Op("pd.transpose");
    src.Tensor("transpose_1_out") = transpose_1(src.Tensor("reshape_1_out"));
    const auto &full_1 =
        src.Op("pd.full", {{"value", src.Attr("full_1_value")}});
    const auto &scale = src.Op("pd.scale");
    src.Tensor("scale_out") = scale(src.Tensor("transpose_1_out"), full_1());

    // The second path to matmul (k).
    const auto &matmul_2 =
        src.Op("pd.matmul",
               {{"transpose_x", src.Attr("matmul_2_transpose_x")},
                {"transpose_y", src.Attr("matmul_2_transpose_y")}});
    src.Tensor("matmul_2_out") =
        matmul_2(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_2_in_2"));
    const auto &add_2 = src.Op("pd.add");
    src.Tensor("add_2_out") =
        add_2(src.Tensor("matmul_2_out"), src.Tensor("add_2_in_2"));
    const auto &full_int_array_2 = src.Op("pd.full_int_array");
    const auto &reshape_2 = src.Op("pd.reshape");
    reshape_2({&src.Tensor("add_2_out"), &full_int_array_2()},
              {&src.Tensor("reshape_2_out"), &src.Tensor("reshape_2_xshape")});
    const auto &transpose_2 = src.Op("pd.transpose");
    src.Tensor("transpose_2_out") = transpose_2(src.Tensor("reshape_2_out"));

    // The third path to matmul (v).
    const auto &matmul_3 =
        src.Op("pd.matmul",
               {{"transpose_x", src.Attr("matmul_3_transpose_x")},
                {"transpose_y", src.Attr("matmul_3_transpose_y")}});
    src.Tensor("matmul_3_out") =
        matmul_3(src.Tensor("matmul_1_in_1"), src.Tensor("matmul_3_in_2"));
    const auto &add_3 = src.Op("pd.add");
    src.Tensor("add_3_out") =
        add_3(src.Tensor("matmul_3_out"), src.Tensor("add_3_in_2"));
    const auto &full_int_array_3 = src.Op("pd.full_int_array");
    const auto &reshape_3 = src.Op("pd.reshape");
    reshape_3({&src.Tensor("add_3_out"), &full_int_array_3()},
              {&src.Tensor("reshape_3_out"), &src.Tensor("reshape_3_xshape")});
    const auto &transpose_3 = src.Op("pd.transpose");
    src.Tensor("transpose_3_out") = transpose_3(src.Tensor("reshape_3_out"));

    // softmax(qk)v
    const auto &matmul_4 =
        src.Op("pd.matmul",
               {{"transpose_x", src.Attr("matmul_4_transpose_x")},
                {"transpose_y", src.Attr("matmul_4_transpose_y")}});
    src.Tensor("matmul_4_out") =
        matmul_4(src.Tensor("scale_out"), src.Tensor("transpose_2_out"));
    const auto &add_4 = src.Op("pd.add");
    src.Tensor("add_4_out") =
        add_4(src.Tensor("matmul_4_out"), src.Tensor("add_4_in_2"));
    const auto &softmax =
        src.Op("pd.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("add_4_out"));
    const auto &matmul_5 =
        src.Op("pd.matmul",
               {{"transpose_x", src.Attr("matmul_5_transpose_x")},
                {"transpose_y", src.Attr("matmul_5_transpose_y")}});
    src.Tensor("matmul_5_out") =
        matmul_5(src.Tensor("softmax_out"), src.Tensor("transpose_3_out"));
    const auto &transpose_4 = src.Op("pd.transpose");
    src.Tensor("transpose_4_out") = transpose_4(src.Tensor("matmul_5_out"));
    const auto &full_int_array_4 = src.Op("pd.full_int_array");
    const auto &reshape_4 = src.Op("pd.reshape");
    reshape_4({&src.Tensor("transpose_4_out"), &full_int_array_4()},
              {&src.Tensor("reshape_4_out"), &src.Tensor("reshape_4_xshape")});

    //
    // Constraints.
    //
    src.RequireNativeCall([](const ir::drr::MatchContext &match_ctx) -> bool {
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

      return true;
    });

    //
    // Result Pattern.
    //
    ir::drr::ResultPattern res = src.ResultPattern();
    // W combine.
    const auto &combine_1 = res.Op("builtin.combine");
    combine_1({&res.Tensor("matmul_1_in_2"),
               &res.Tensor("matmul_2_in_2"),
               &res.Tensor("matmul_3_in_2")},
              {&res.Tensor("combine_1_out")});
    const auto &concat_axis = res.Attr(
        [](const ir::drr::MatchContext &match_ctx) -> int { return 0; });
    const auto &concat_1 = res.Op("pd.concat", {{"axis", concat_axis}});
    res.Tensor("concat_1_out") = concat_1(res.Tensor("combine_1_out"));
    const auto &reshape_5_shape = res.Attr(
        [](const ir::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto matmul_1_in_2 = match_ctx.Tensor("matmul_1_in_2").Shape();
          return {-1, 3, matmul_1_in_2.at(1)};
        });
    const auto &full_int_array_5 =
        res.Op("pd.full_int_array", {{"value", reshape_5_shape}});
    const auto &reshape_5 = res.Op("pd.reshape");
    reshape_5({&res.Tensor("concat_1_out"), &full_int_array_5()},
              {&res.Tensor("reshape_5_out"), &res.NoneTensor()});

    // Bias combine.
    const auto &combine_2 = res.Op("builtin.combine");
    combine_2({&res.Tensor("add_1_in_2"),
               &res.Tensor("add_2_in_2"),
               &res.Tensor("add_3_in_2")},
              {&res.Tensor("combine_2_out")});
    const auto &concat_2 = res.Op("pd.concat", {{"axis", concat_axis}});
    res.Tensor("concat_2_out") = concat_2(res.Tensor("combine_2_out"));
    const auto &reshape_6_shape = res.Attr(
        [](const ir::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          //   std::vector<int64_t> new_shape;
          //   auto add_1_in_2_shape = match_ctx.Tensor("add_1_in_2").Shape();
          return {3, -1};
        });
    const auto &full_int_array_6 =
        res.Op("pd.full_int_array", {{"value", reshape_6_shape}});
    const auto &reshape_6 = res.Op("pd.reshape");
    reshape_6({&res.Tensor("concat_2_out"), &full_int_array_6()},
              {&res.Tensor("reshape_6_out"), &res.NoneTensor()});

    const auto &head_number =
        res.Attr([](const ir::drr::MatchContext &match_ctx) -> int {
          const auto &full_int_array_1_value =
              match_ctx.Attr<std::vector<int64_t>>("full_int_array_1_value");
          return full_int_array_1_value.at(2);
        });
    const auto &alpha =
        res.Attr([](const ir::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("full_1_value");
        });
    const auto &multihead_matmul =
        res.Op("pd.multihead_matmul",
               {{"head_number", head_number}, {"alpha", alpha}});
    multihead_matmul({&res.Tensor("matmul_1_in_1"),
                      &res.Tensor("reshape_5_out"),
                      &res.Tensor("reshape_6_out"),
                      &res.Tensor("add_4_in_2")},
                     {&res.Tensor("reshape_4_out")});
  }
};

class AttentionFusePass : public ir::Pass {
 public:
  AttentionFusePass() : ir::Pass("AttentionFusePass", 1) {}

  bool Initialize(ir::IrContext *context) override {
    ir::RewritePatternSet ps(context);
    ps.Add(MultiHeadMatmulFusePattern().Build(context));
    // Add other attention variant fuse pattern.

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

void BuildProgram(ir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp matmul_1_in_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 300, 256},
                                             0.9,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  // The first path to matmul with scale (q).
  paddle::dialect::FullOp matmul_1_in_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::MatmulOp matmul_1 = builder.Build<paddle::dialect::MatmulOp>(
      matmul_1_in_1.out(), matmul_1_in_2.out(), false, false);

  paddle::dialect::FullOp add_1_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::AddOp add_1 =
      builder.Build<paddle::dialect::AddOp>(matmul_1.out(), add_1_in_2.out());

  paddle::dialect::ReshapeOp reshape_1 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_1.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_1 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_1.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  paddle::dialect::ScaleOp scale_op = builder.Build<paddle::dialect::ScaleOp>(
      transpose_1.out(), 0.1767766922712326, 0.0, true);

  // The second path to matmul (k).
  paddle::dialect::FullOp matmul_2_in_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::MatmulOp matmul_2 = builder.Build<paddle::dialect::MatmulOp>(
      matmul_1_in_1.out(), matmul_2_in_2.out(), false, false);

  paddle::dialect::FullOp add_2_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::AddOp add_op2 =
      builder.Build<paddle::dialect::AddOp>(matmul_2.out(), add_2_in_2.out());

  paddle::dialect::ReshapeOp reshape_2 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_op2.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_2 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_2.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  // The third path to matmul (v).
  paddle::dialect::FullOp full_mat1_y_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(
          matmul_1_in_1.out(), full_mat1_y_op.out(), false, false);

  paddle::dialect::FullOp add_3_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::AddOp add_3 =
      builder.Build<paddle::dialect::AddOp>(matmul_op1.out(), add_3_in_2.out());

  paddle::dialect::ReshapeOp reshape_3 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_3.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_3 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_3.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  // softmax(qk)v
  paddle::dialect::MatmulOp matmul_4 = builder.Build<paddle::dialect::MatmulOp>(
      scale_op.out(), transpose_2.out(), false, true);

  paddle::dialect::FullOp add_4_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 8, 300, 300},
      1.5,
      phi::DataType::FLOAT32,
      phi::CPUPlace());

  paddle::dialect::AddOp add_4 =
      builder.Build<paddle::dialect::AddOp>(matmul_4.out(), add_4_in_2.out());

  paddle::dialect::SoftmaxOp softmax_op =
      builder.Build<paddle::dialect::SoftmaxOp>(add_4.out(), -1);
  paddle::dialect::MatmulOp matmul_op5 =
      builder.Build<paddle::dialect::MatmulOp>(
          softmax_op.out(), transpose_3.out(), false, false);

  paddle::dialect::TransposeOp transpose_4 =
      builder.Build<paddle::dialect::TransposeOp>(matmul_op5.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  paddle::dialect::ReshapeOp reshape_4 =
      builder.Build<paddle::dialect::ReshapeOp>(
          transpose_4.out(), std::vector<int64_t>{0, 0, 256});

  builder.Build<paddle::dialect::FetchOp>(reshape_4.out(), "out", 0);
}

TEST(DrrTest, AttentionFuse) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder(ctx, program.block());
  BuildProgram(builder);
  EXPECT_EQ(program.block()->size(), 33u);

  ir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<AttentionFusePass>());
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
  EXPECT_EQ(program.block()->size(), 20u);
}
