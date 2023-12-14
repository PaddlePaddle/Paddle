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
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/pass/pass_manager.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(shape_optimization, shape_optimization_pass) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);

  pir::Operation *op0 = test::CreateDenseTensorOp(
      ctx, {-1, 2}, {"op0_attr"}, {"create_dense_tensor_op0"});
  pir::Operation *op1 = test::CreateDenseTensorOp(
      ctx, {-1, 2, 2}, {"op1_attr"}, {"create_dense_tensor_op1"});
  program.block()->push_back(op0);
  program.block()->push_back(op1);

  EXPECT_EQ(program.block()->size(), 2u);

  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  // 5 ConstantOp + 5 TensorDim + 2 TieShape + op0 + op1 + 1 funcOp == 15 Ops.
  EXPECT_EQ(program.block()->size(), 2u);

  pir::SymbolicDimMgr mgr(program.module_op());
  EXPECT_TRUE(mgr.Load());
  EXPECT_TRUE(mgr.Save());
}

TEST(shape_optimization, expand_shape_of_op_pattern) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::Operation *op0 =
      test::CreateDenseTensorOp(ctx,
                                {pir::ShapedTypeInterface::kDynamic, 2, 2},
                                {"op1_0ttr"},
                                {"create_dense_tensor_op0"});
  program.block()->push_back(op0);
  builder.Build<pir::shape::ShapeOfOp>(op0->result(0));

  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  pir::SymbolicDimMgr mgr(program.module_op());
  EXPECT_TRUE(mgr.Load());
  EXPECT_TRUE(mgr.Save());
}

TEST(shape_optimization, dim_of_shaped_type_op_interface_pattern) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::Operation *op0 =
      test::CreateDenseTensorOp(ctx,
                                {pir::ShapedTypeInterface::kDynamic, 2},
                                {"op1_0ttr"},
                                {"create_dense_tensor_op0"});
  program.block()->push_back(op0);
  std::vector<int> perm = {1, 0};

  pir::Operation *op1 =
      builder.Build<paddle::dialect::TransposeOp>(op0->result(0), perm);

  builder.Build<pir::shape::ShapeOfOp>(op1->result(0));

  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  pir::SymbolicDimMgr mgr(program.module_op());
  EXPECT_TRUE(mgr.Load());
  EXPECT_TRUE(mgr.Save());
}
