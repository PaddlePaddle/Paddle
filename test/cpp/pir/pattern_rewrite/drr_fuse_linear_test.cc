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
#include "paddle/fluid/pir/transforms/gpu/fused_gemm_epilogue_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 512, 64},
                                             1.5);
  // linear 1
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());
  // linear 2
  paddle::dialect::FullOp full_weight_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             1.5);
  paddle::dialect::FullOp full_bias_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{128}, 1.0);
  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(add_op1.out(),
                                               full_weight_op2.out());
  paddle::dialect::AddOp add_op2 = builder.Build<paddle::dialect::AddOp>(
      matmul_op2.out(), full_bias_op2.out());
  paddle::dialect::ReluOp relu_op =
      builder.Build<paddle::dialect::ReluOp>(add_op2.out());
  // linear 3
  paddle::dialect::FullOp full_weight_op3 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 64},
                                             1.5);
  paddle::dialect::FullOp full_bias_op3 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(relu_op.out(),
                                               full_weight_op3.out());
  paddle::dialect::AddOp add_op3 = builder.Build<paddle::dialect::AddOp>(
      matmul_op3.out(), full_bias_op3.out());
  paddle::dialect::GeluOp gelu_op1 =
      builder.Build<paddle::dialect::GeluOp>(add_op3.out());
  // linear 4
  paddle::dialect::FullOp full_weight_op4 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op4 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op4 =
      builder.Build<paddle::dialect::MatmulOp>(gelu_op1.out(),
                                               full_weight_op4.out());
  paddle::dialect::AddOp add_op4 = builder.Build<paddle::dialect::AddOp>(
      matmul_op4.out(), full_bias_op4.out());
  paddle::dialect::GeluOp gelu_op2 =
      builder.Build<paddle::dialect::GeluOp>(add_op4.out());

  // backward
  paddle::dialect::FullOp full_grad_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 512, 64}, 1.0);

  paddle::dialect::GeluGradOp gelu_op2_grad =
      builder.Build<paddle::dialect::GeluGradOp>(
          add_op4.out(), full_grad_op.out(), false);
  // backward linear 4
  paddle::dialect::AddGradOp add_op4_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op4.out(), full_bias_op4.out(), gelu_op2_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op4_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          gelu_op1.out(), full_weight_op4.out(), add_op4_grad.x_grad());

  paddle::dialect::GeluGradOp gelu_op1_grad =
      builder.Build<paddle::dialect::GeluGradOp>(
          add_op3.out(), matmul_op4_grad.x_grad(), false);
  // backward linear 3
  paddle::dialect::AddGradOp add_op3_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op3.out(), full_bias_op3.out(), gelu_op1_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op3_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          relu_op.out(), full_weight_op3.out(), add_op3_grad.x_grad());

  paddle::dialect::ReluGradOp relu_op_grad =
      builder.Build<paddle::dialect::ReluGradOp>(relu_op.out(),
                                                 matmul_op3_grad.x_grad());
  // backward linear 2
  paddle::dialect::AddGradOp add_op2_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op2.out(), full_bias_op2.out(), relu_op_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op2_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          add_op1.out(), full_weight_op2.out(), add_op2_grad.x_grad());
  // backward linear 1
  paddle::dialect::AddGradOp add_op1_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op1.out(), full_bias_op1.out(), matmul_op2_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op1_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          full_input_op1.out(), full_weight_op1.out(), add_op1_grad.x_grad());

  builder.Build<paddle::dialect::FetchOp>(gelu_op2.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(matmul_op1_grad.x_grad(), "dx", 1);
}

TEST(DrrTest, FusedLinear) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 34u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedGemmEpiloguePass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Unavailable("pm fail to run program"));
  EXPECT_EQ(program.block()->size(), 22u);
}
