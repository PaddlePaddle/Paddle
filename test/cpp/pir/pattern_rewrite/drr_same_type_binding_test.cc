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
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

/* Source pattern:
                                       input1
                                    /  |  \  \  \
                                  /    |   \   \    \
             full               /      |    |    \     \           full_tmp
            /  |        transpose1      | trans2 trans3    \        /   |
           /   |         /    |        |    |      |        \      /    |
    softmax1   |        /     |        |    |      |          \   /     |
         \     |      /    softmax2    |    |      |          add1      |
           \   |    /             \    |     \    /             |       |
           layernorm             matmul2     matmul1             \      |
             / | \                   |         |                  \     |
           /   |   \                  \       /                     \   |
         /     |     \                 matmul3                        add2
        |      |      |                /  |  \                          |
        |      |      |              /    |    \                        |
        |      |      |            /      |      \                      |
        |      |      |         trans4  trans5  trans6                  |
        |      |      |           |       |        |                    |
        |      |      |         relu1  softmax3 softmax4              relu2
        |      |      |           |       |        |                    |
    output0 output1 output2    output3  output4  output5             output6
*/

// This class is for test cases of the same type of OP.
// (without considering the computational logic between OPs,
// only focusing on the process of matching and replacing)
class SameTypeBindingTestPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SameTypeBindingTestPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();

    // path 1
    const auto &transpose_1 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_1")}});
    src.Tensor("transpose_1_out") = transpose_1(src.Tensor("input_1"));
    const auto &softmax_2 =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_2_axis")}});
    src.Tensor("softmax_2_out") = softmax_2(src.Tensor("transpose_1_out"));
    const auto &matmul_2 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_2_transpose_x")},
                {"transpose_y", src.Attr("matmul_2_transpose_y")}});
    src.Tensor("matmul_2_out") =
        matmul_2(src.Tensor("softmax_2_out"), src.Tensor("input_1"));

    // path 2
    const auto &full_1 = src.Op("pd_op.full",
                                {{"shape", src.Attr("shape_1")},
                                 {"value", src.Attr("value_1")},
                                 {"dtype", src.Attr("dtype_1")},
                                 {"place", src.Attr("place_1")}});
    src.Tensor("full_1_out") = full_1();
    const auto &softmax_1 =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_1_axis")}});
    src.Tensor("softmax_1_out") = softmax_1(src.Tensor("full_1_out"));
    const auto &layernorm_1 =
        src.Op("pd_op.layer_norm",
               {{"epsilon", src.Attr("layernorm_epsilon")},
                {"begin_norm_axis", src.Attr("layernorm_begin_norm_axis")}});
    layernorm_1({&src.Tensor("transpose_1_out"),
                 &src.Tensor("full_1_out"),
                 &src.Tensor("softmax_1_out")},
                {&src.Tensor("output0"),
                 &src.Tensor("output1"),
                 &src.Tensor("output2")});

    // path 3
    const auto &transpose_2 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_2")}});
    const auto &transpose_3 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_3")}});
    const auto &matmul_1 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_1_transpose_x")},
                {"transpose_y", src.Attr("matmul_1_transpose_y")}});
    src.Tensor("matmul_1_out") = matmul_1(transpose_2(src.Tensor("input_1")),
                                          transpose_3(src.Tensor("input_1")));
    const auto &matmul_3 =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_3_transpose_x")},
                {"transpose_y", src.Attr("matmul_3_transpose_y")}});
    src.Tensor("matmul_3_out") =
        matmul_3(src.Tensor("matmul_2_out"), src.Tensor("matmul_1_out"));
    const auto &transpose_4 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_4")}});
    const auto &transpose_5 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_5")}});
    const auto &transpose_6 =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm_6")}});
    const auto &relu_1 = src.Op("pd_op.relu");
    const auto &softmax_3 =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_3_axis")}});
    const auto &softmax_4 =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_4_axis")}});
    src.Tensor("output3") = relu_1(transpose_4(src.Tensor("matmul_3_out")));
    src.Tensor("output4") = softmax_3(transpose_5(src.Tensor("matmul_3_out")));
    src.Tensor("output5") = softmax_4(transpose_6(src.Tensor("matmul_3_out")));

    // path 4
    const auto &full_tmp = src.Op("pd_op.full",
                                  {{"shape", src.Attr("shape_tmp")},
                                   {"value", src.Attr("value_tmp")},
                                   {"dtype", src.Attr("dtype_tmp")},
                                   {"place", src.Attr("place_tmp")}});
    src.Tensor("full_tmp_out") = full_tmp();
    const auto &add_1 = src.Op("pd_op.add");
    src.Tensor("add_1_out") =
        add_1(src.Tensor("input_1"), src.Tensor("full_tmp_out"));
    const auto &add_2 = src.Op("pd_op.add");
    src.Tensor("add_2_out") =
        add_2(src.Tensor("add_1_out"), src.Tensor("full_tmp_out"));
    const auto &relu_2 = src.Op("pd_op.relu");
    src.Tensor("output6") = relu_2(src.Tensor("add_2_out"));

    paddle::drr::ResultPattern res = src.ResultPattern();
    const auto &transpose_7 =
        res.Op("pd_op.transpose", {{"perm", src.Attr("perm_4")}});
    res.Tensor("output0") = transpose_7(res.Tensor("input_1"));
    const auto &transpose_8 =
        res.Op("pd_op.transpose", {{"perm", src.Attr("perm_5")}});
    res.Tensor("output1") = transpose_8(res.Tensor("input_1"));
    const auto &full_2 = res.Op("pd_op.full",
                                {{"shape", src.Attr("shape_tmp")},
                                 {"value", src.Attr("value_tmp")},
                                 {"dtype", src.Attr("dtype_tmp")},
                                 {"place", src.Attr("place_tmp")}});
    const auto &full_3 = res.Op("pd_op.full",
                                {{"shape", src.Attr("shape_tmp")},
                                 {"value", src.Attr("value_tmp")},
                                 {"dtype", src.Attr("dtype_tmp")},
                                 {"place", src.Attr("place_tmp")}});
    const auto &full_4 = res.Op("pd_op.full",
                                {{"shape", src.Attr("shape_tmp")},
                                 {"value", src.Attr("value_tmp")},
                                 {"dtype", src.Attr("dtype_tmp")},
                                 {"place", src.Attr("place_tmp")}});
    const auto &full_5 = res.Op("pd_op.full",
                                {{"shape", src.Attr("shape_tmp")},
                                 {"value", src.Attr("value_tmp")},
                                 {"dtype", src.Attr("dtype_tmp")},
                                 {"place", src.Attr("place_tmp")}});
    const auto &full_6 = res.Op("pd_op.full",
                                {{"shape", src.Attr("shape_tmp")},
                                 {"value", src.Attr("value_tmp")},
                                 {"dtype", src.Attr("dtype_tmp")},
                                 {"place", src.Attr("place_tmp")}});
    res.Tensor("output2") = full_2();
    res.Tensor("output3") = full_3();
    res.Tensor("output4") = full_4();
    res.Tensor("output5") = full_5();
    res.Tensor("output6") = full_6();
  }
};

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  // path 1
  paddle::dialect::TransposeOp transpose_op1 =
      builder.Build<paddle::dialect::TransposeOp>(full_input_op1.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::SoftmaxOp softmax_op2 =
      builder.Build<paddle::dialect::SoftmaxOp>(transpose_op1.out(), -1);

  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(softmax_op2.out(),
                                               full_input_op1.out());

  // path 2
  paddle::dialect::FullOp full_op_scale =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{48},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  paddle::dialect::SoftmaxOp softmax_op_bias =
      builder.Build<paddle::dialect::SoftmaxOp>(full_op_scale.out(), -1);
  paddle::dialect::LayerNormOp layernorm_op1 =
      builder.Build<paddle::dialect::LayerNormOp>(
          transpose_op1.out(), full_op_scale.out(), softmax_op_bias.out());

  // path 3
  paddle::dialect::TransposeOp transpose_op2 =
      builder.Build<paddle::dialect::TransposeOp>(full_input_op1.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::TransposeOp transpose_op3 =
      builder.Build<paddle::dialect::TransposeOp>(full_input_op1.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(transpose_op2.out(),
                                               transpose_op3.out());

  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(matmul_op2.out(),
                                               matmul_op1.out());

  paddle::dialect::TransposeOp transpose_op4 =
      builder.Build<paddle::dialect::TransposeOp>(matmul_op3.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::ReluOp relu_op1 =
      builder.Build<paddle::dialect::ReluOp>(transpose_op4.out());

  paddle::dialect::TransposeOp transpose_op5 =
      builder.Build<paddle::dialect::TransposeOp>(matmul_op3.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::SoftmaxOp softmax_op3 =
      builder.Build<paddle::dialect::SoftmaxOp>(transpose_op5.out(), -1);

  paddle::dialect::TransposeOp transpose_op6 =
      builder.Build<paddle::dialect::TransposeOp>(matmul_op3.out(),
                                                  std::vector<int>{0, 1, 2});

  paddle::dialect::SoftmaxOp softmax_op4 =
      builder.Build<paddle::dialect::SoftmaxOp>(transpose_op6.out(), -1);

  // path 4
  paddle::dialect::FullOp full_input_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      full_input_op1.out(), full_input_op2.out());

  paddle::dialect::AddOp add_op2 = builder.Build<paddle::dialect::AddOp>(
      add_op1.out(), full_input_op2.out());

  paddle::dialect::ReluOp relu_op2 =
      builder.Build<paddle::dialect::ReluOp>(add_op2.out());

  // tail
  paddle::dialect::MatmulOp matmul_op4 =
      builder.Build<paddle::dialect::MatmulOp>(layernorm_op1.variance(),
                                               layernorm_op1.mean());

  paddle::dialect::MatmulOp matmul_op5 =
      builder.Build<paddle::dialect::MatmulOp>(relu_op1.out(),
                                               softmax_op3.out());

  paddle::dialect::MatmulOp matmul_op6 =
      builder.Build<paddle::dialect::MatmulOp>(softmax_op4.out(),
                                               relu_op2.out());

  builder.Build<paddle::dialect::FetchOp>(matmul_op4.out(), "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(matmul_op5.out(), "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(matmul_op6.out(), "out3", 2);
}

class DrrPatternRewritePass : public pir::PatternRewritePass {
 public:
  DrrPatternRewritePass()
      : pir::PatternRewritePass("drr_pattern_rewrite_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<SameTypeBindingTestPattern>(context));

    return ps;
  }
};

TEST(DrrTest, drr_demo) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 27u);

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<DrrPatternRewritePass>());
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Unavailable("pm fail to run program"));
  EXPECT_EQ(program.block()->size(), 13u);
}
