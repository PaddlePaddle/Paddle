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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 0.5);

  paddle::dialect::FullOp weight_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 0.5);
  paddle::dialect::FullOp weight_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 64}, 0.5);
  paddle::dialect::FullOp weight_3 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{32, 128}, 0.5);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_1.out());
  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_2.out());
  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_3.out());

  builder.Build<paddle::dialect::FetchOp>(matmul_op1.out(), "x", 0);
  builder.Build<paddle::dialect::FetchOp>(matmul_op2.out(), "y", 1);
  builder.Build<paddle::dialect::FetchOp>(matmul_op3.out(), "z", 1);
}

TEST(Cinn, FuseMatmul) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);
  ASSERT_EQ((program.block()->size()), 10u);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreateFuseParallelMatmulPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  ASSERT_EQ((pm.Run(&program)), true);
  ASSERT_EQ((program.block()->size()), 20u);
}

// [64, 32] * [16, 32, 32] => [16, 64, 32]
void BuildBatchProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 32}, 0.5);

  paddle::dialect::FullOp weight_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{16, 32, 32}, 0.5);
  paddle::dialect::FullOp weight_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{16, 32, 64}, 0.5);
  paddle::dialect::FullOp weight_3 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{16, 32, 128}, 0.5);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_1.out());
  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_2.out());
  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(x.out(), weight_3.out());

  builder.Build<paddle::dialect::FetchOp>(matmul_op1.out(), "x", 0);
  builder.Build<paddle::dialect::FetchOp>(matmul_op2.out(), "y", 1);
  builder.Build<paddle::dialect::FetchOp>(matmul_op3.out(), "z", 1);
}

TEST(Cinn, FuseBatchMatmul) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildBatchProgram(builder);
  ASSERT_EQ((program.block()->size()), 10u);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreateFuseParallelMatmulPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  ASSERT_EQ((pm.Run(&program)), true);
  ASSERT_EQ((program.block()->size()), 20u);
}
