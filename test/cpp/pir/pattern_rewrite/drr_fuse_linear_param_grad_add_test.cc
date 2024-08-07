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
#include "paddle/fluid/pir/transforms/gpu/fused_linear_param_grad_add_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram0(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32}, 1.0);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());

  paddle::dialect::FullOp full_d_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::FullOp full_d_out_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::AddGradOp add_grad_op1 =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op1.out(), full_bias_op1.out(), full_d_out_op1.out());

  paddle::dialect::MatmulGradOp matmul_grad_op1 =
      builder.Build<paddle::dialect::MatmulGradOp>(
          full_input_op1.out(), full_weight_op1.out(), add_grad_op1.x_grad());

  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      full_d_weight_op1.out(), matmul_grad_op1.y_grad());

  builder.Build<paddle::dialect::FetchOp>(add_op1.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(add_grad_op1.y_grad(), "dbias", 1);
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dweight", 2);
}

void BuildProgram1(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());

  paddle::dialect::FullOp full_d_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::FullOp full_d_out_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::MatmulGradOp matmul_grad_op1 =
      builder.Build<paddle::dialect::MatmulGradOp>(
          full_input_op1.out(), full_weight_op1.out(), full_d_out_op1.out());

  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      full_d_weight_op1.out(), matmul_grad_op1.y_grad());

  builder.Build<paddle::dialect::FetchOp>(matmul_op1.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dweight", 1);
}

void BuildProgram2(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());

  paddle::dialect::FullOp full_d_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::FullOp full_d_out_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(
          full_input_op1.out(), full_d_out_op1.out(), true, false);

  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      full_d_weight_op1.out(), matmul_op2.out());

  builder.Build<paddle::dialect::FetchOp>(matmul_op1.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dweight", 1);
}

void BuildProgram3(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32}, 1.0);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());

  paddle::dialect::FullOp full_d_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::FullOp full_d_out_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::AddGradOp add_grad_op1 =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op1.out(), full_bias_op1.out(), full_d_out_op1.out());

  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(
          add_grad_op1.x_grad(), full_weight_op1.out(), false, true);

  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(
          full_input_op1.out(), add_grad_op1.x_grad(), true, false);

  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      full_d_weight_op1.out(), matmul_op3.out());

  builder.Build<paddle::dialect::FetchOp>(add_op1.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(add_grad_op1.y_grad(), "dbias", 1);
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dweight", 2);
  builder.Build<paddle::dialect::FetchOp>(matmul_op2.out(), "dx", 3);
}

void BuildProgram4(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp input_x = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2048, 5120}, 1.5, phi::DataType::FLOAT16);
  paddle::dialect::FullOp input_dy = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2048, 6912}, 2.5, phi::DataType::FLOAT16);
  paddle::dialect::FullOp input_weight_grad =
      builder.Build<paddle::dialect::FullOp>(
          std::vector<int64_t>{5120, 6912}, 1.0, phi::DataType::FLOAT16);

  paddle::dialect::ReshapeOp reshape_x =
      builder.Build<paddle::dialect::ReshapeOp>(
          input_x.out(), std::vector<int64_t>{4096, 5120});
  paddle::dialect::ReshapeOp reshape_dy =
      builder.Build<paddle::dialect::ReshapeOp>(
          input_dy.out(), std::vector<int64_t>{4096, 6912});
  paddle::dialect::MatmulOp matmul_op =
      builder.Build<paddle::dialect::MatmulOp>(
          reshape_x.out(), reshape_dy.out(), true, false);
  paddle::dialect::ReshapeOp output_dw =
      builder.Build<paddle::dialect::ReshapeOp>(
          matmul_op.out(), std::vector<int64_t>{5120, 6912});
  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      input_weight_grad.out(), output_dw.out());
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dw", 0);
}

bool verify_pass(const pir::Program &program) {
  for (auto &op : *(program.block())) {
    if (op.name() == paddle::dialect::FusedLinearParamGradAddOp::name()) {
      return true;
    }
  }
  return false;
}

TEST(DrrTest, FusedLinearParamGradAdd0) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram0(builder);

  EXPECT_EQ(program.block()->size(), 13u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedLinearParamGradAddPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::InvalidArgument(
                        "Required pm.Run(&program) should be true"));
  EXPECT_EQ(verify_pass(program), true);
}

TEST(DrrTest, FusedLinearParamGradAdd1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram1(builder);

  EXPECT_EQ(program.block()->size(), 9u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedLinearParamGradAddPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::InvalidArgument(
                        "Required pm.Run(&program) should be true"));
  EXPECT_EQ(verify_pass(program), true);
}

TEST(DrrTest, FusedLinearParamGradAdd2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram2(builder);

  EXPECT_EQ(program.block()->size(), 9u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedLinearParamGradAddPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::InvalidArgument(
                        "Required pm.Run(&program) should be true"));
  EXPECT_EQ(verify_pass(program), true);
}

TEST(DrrTest, FusedLinearParamGradAdd3) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram3(builder);

  EXPECT_EQ(program.block()->size(), 15u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedLinearParamGradAddPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::InvalidArgument(
                        "Required pm.Run(&program) should be true"));
  EXPECT_EQ(verify_pass(program), true);
}

TEST(DrrTest, FusedMatmulReshapeMatmulAddPattern) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram4(builder);

  EXPECT_EQ(program.block()->size(), 12u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateFusedLinearParamGradAddPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::InvalidArgument(
                        "Required pm.Run(&program) should be true"));
  EXPECT_EQ(verify_pass(program), true);
  EXPECT_EQ(program.block()->size(), 5u);
}
