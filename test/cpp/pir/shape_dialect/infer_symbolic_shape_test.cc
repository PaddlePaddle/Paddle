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

#include <gtest/gtest.h>
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/pass/pass_manager.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(infer_symbolic_shape, op_with_same_operands_and_result_shape) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::Builder builder = ::pir::Builder(ctx, program.block());
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  // Build Ops
  auto data_op_x = builder.Build<paddle::dialect::DataOp>(
      "x",
      std::vector<int64_t>({1, 64, -1, -1}),
      phi::DataType::FLOAT32,
      phi::Place());

  auto abs_op = builder.Build<paddle::dialect::AbsOp>(data_op_x.result(0));
  auto cast_op = builder.Build<paddle::dialect::CastOp>(abs_op.result(0),
                                                        phi::DataType::FLOAT64);
  auto scale_op = builder.Build<paddle::dialect::ScaleOp>(
      cast_op.result(0), 1.0, 0.0, true);
  auto relu_op = builder.Build<paddle::dialect::ReluOp>(scale_op.result(0));

  // Run shape optimization pass
  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  // Check the results
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(&program);

  pir::Value abs_op_res = abs_op.result(0);
  pir::Value cast_op_res = cast_op.result(0);
  pir::Value scale_op_res = scale_op.result(0);
  pir::Value relu_op_res = relu_op.result(0);

  // All op's symbolic result should be [1,  64,  S0, S1]
  std::vector<symbol::DimExpr> dims_ref{symbol::DimExpr(1),
                                        symbol::DimExpr(64),
                                        symbol::DimExpr("S0"),
                                        symbol::DimExpr("S1")};
  symbol::ShapeOrDataDimExprs shape_ref{
      symbol::TensorShapeOrDataDimExprs(dims_ref)};

  EXPECT_EQ(shape_analysis.GetShapeOrDataForValue(abs_op_res), shape_ref);
  EXPECT_TRUE(shape_analysis.IsShapeEqual(abs_op_res, cast_op_res));
  EXPECT_TRUE(shape_analysis.IsShapeEqual(abs_op_res, scale_op_res));
  EXPECT_TRUE(shape_analysis.IsShapeEqual(abs_op_res, relu_op_res));
}
