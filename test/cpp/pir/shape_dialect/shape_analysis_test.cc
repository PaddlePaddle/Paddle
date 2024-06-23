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
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(shape_optimization, shape_optimization_pass) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::Builder builder = ::pir::Builder(ctx, program.block());
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  // Build Ops
  auto data_op_x = builder.Build<paddle::dialect::DataOp>(
      "x",
      std::vector<int64_t>({1, 64, -1, 64, 2}),
      phi::DataType::FLOAT32,
      phi::Place());
  auto abs_op = builder.Build<paddle::dialect::AbsOp>(data_op_x.result(0));

  auto data_op_y = builder.Build<paddle::dialect::DataOp>(
      "y",
      std::vector<int64_t>({-1, 128, 32, 2, 2}),
      phi::DataType::FLOAT32,
      phi::Place());

  auto data_op_z = builder.Build<paddle::dialect::DataOp>(
      "z",
      std::vector<int64_t>({128, 32, 2, 2}),
      phi::DataType::FLOAT32,
      phi::Place());

  auto data_op_w = builder.Build<paddle::dialect::DataOp>(
      "w",
      std::vector<int64_t>({64, 32, 2, 2, 2}),
      phi::DataType::FLOAT32,
      phi::Place());

  auto data_op_u = builder.Build<paddle::dialect::DataOp>(
      "u", std::vector<int64_t>({-1, 5}), phi::DataType::FLOAT32, phi::Place());

  auto relu_op = builder.Build<paddle::dialect::ReluOp>(data_op_u.result(0));

  auto data_op_v = builder.Build<paddle::dialect::DataOp>(
      "v", std::vector<int64_t>({5, 2}), phi::DataType::FLOAT32, phi::Place());

  auto data_op_p = builder.Build<paddle::dialect::DataOp>(
      "p", std::vector<int64_t>({5, 2}), phi::DataType::FLOAT32, phi::Place());

  auto data_op_q = builder.Build<paddle::dialect::DataOp>(
      "q", std::vector<int64_t>({-1, 2}), phi::DataType::FLOAT32, phi::Place());

  auto addmm_op1 = builder.Build<paddle::dialect::AddmmOp>(
      data_op_q.result(0), data_op_u.result(0), data_op_v.result(0));

  auto addmm_op2 = builder.Build<paddle::dialect::AddmmOp>(
      data_op_q.result(0), relu_op.result(0), data_op_p.result(0));

  // Run shape optimization pass
  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  // Check the results
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(&program);

  pir::Value data_op_x_res = data_op_x.result(0);
  pir::Value abs_op_res = abs_op.result(0);
  pir::Value data_op_y_res = data_op_y.result(0);
  pir::Value data_op_z_res = data_op_z.result(0);
  pir::Value data_op_w_res = data_op_w.result(0);
  pir::Value addmm_op1_res = addmm_op1.result(0);
  pir::Value addmm_op2_res = addmm_op2.result(0);

  // data_op_x_res    [  1,  64,  S0, 64, 2]
  // abs_op_res       [  1,  64,  S0, 64, 2]
  // data_op_y_res    [ S1,  128, 32,  2, 2]
  // data_op_z_res    [128,   32,  2,  2   ]
  // data_op_w_res    [ 64,   32,  2,  2, 2]
  EXPECT_TRUE(shape_analysis.IsShapeEqual(data_op_x_res, abs_op_res));
  EXPECT_TRUE(shape_analysis.IsSameNumel(data_op_x_res, abs_op_res));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(data_op_x_res, data_op_y_res));
  EXPECT_FALSE(shape_analysis.IsSameNumel(data_op_x_res, data_op_y_res));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(data_op_z_res, data_op_w_res));
  EXPECT_TRUE(shape_analysis.IsSameNumel(data_op_z_res, data_op_w_res));
  EXPECT_TRUE(shape_analysis.IsProductEqual(
      data_op_x_res, {0, 2, 4}, abs_op_res, {0, 2, 4}));
  EXPECT_TRUE(
      shape_analysis.IsProductEqual(data_op_x_res, 1, 4, abs_op_res, 1, 4));
  EXPECT_TRUE(shape_analysis.IsProductEqual(
      data_op_z_res, {0, 1}, data_op_w_res, {0, 1, 3}));
  EXPECT_TRUE(
      shape_analysis.IsProductEqual(data_op_z_res, 0, 3, data_op_w_res, 0, 4));
  EXPECT_FALSE(shape_analysis.IsProductEqual(
      data_op_x_res, {0, 1, 2}, data_op_y_res, {0, 1, 2}));
  EXPECT_TRUE(shape_analysis.IsProductEqual(
      data_op_x_res, {1, 3}, data_op_y_res, {1, 2}));
  EXPECT_TRUE(
      shape_analysis.IsProductEqual(data_op_x_res, 3, 5, data_op_y_res, 1, 2));

  // op share cache
  EXPECT_TRUE(shape_analysis.IsShapeEqual(addmm_op1_res, addmm_op2_res));
}
