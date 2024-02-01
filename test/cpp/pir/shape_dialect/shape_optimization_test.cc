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
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/pass/pass_manager.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

// This is a simple net for shape optimization pass test with
// reshape, elementwise, slice, broadcast OPs included.

// clang-format off
/*  ================================== Unary Op ==================================
 *   x=pd.data(x)                 x=pd.abs(x)
 * x —-----------> [1  64  S0 1] —-----------> [1  64  S0 1]
 *
 *
 *                                y=pd.reshape(y)                                    y=pd.cast(y)
 *    y=pd.data(y)              shape={1, 64, -1, 2}                                 （simplify）
 * y —-----------> [S1 128 32] --------------—-------> [1  64 (S1*128*32)/1/64/2 2] ------------> [1  64 (S1*32) 2]
 *
 *
 *                                     z=pd.slice(z)
 *                                 axes   = [0, 1, 2, 3]
 *                                 starts = [2, 2, 2, 2]
 *    z=pd.data(z)                 ends   = [-2,-2,-2,-2]                            z=pd.relu(z)
 * z —-----------> [S2 S3 S4, S5] -------------—---------> [S2-4, S3-4, S4-4, S5-4] --------------> [S2-4, S3-4, S4-4, S5-4]
 *
 * ================================== Binary OP ==================================
 *                                    res = x + y (Maybe Broadcast)
 * [1  64  S0 1] + [1  64 (S1*32) 2] -------------—---------->[1  64  BC(S0,S1*32)  2]
 *
 *
 *                                  res = res - z (Maybe Broadcast)
 * [1  64  BC(S0,S1*32)  2] - [S2-4, S3-4, S4-4, S5-4] = [BC(1,S2-4)  64  BC(BC(S0,S1*32,S4-4)  2]
*/
// clang-format on

TEST(shape_optimization, shape_optimization_pass) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::Builder builder = ::pir::Builder(ctx, program.block());
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  // x related Ops
  auto data_op_x = builder.Build<paddle::dialect::DataOp>(
      "x",
      std::vector<int64_t>({1, 64, -1, 1}),
      phi::DataType::FLOAT32,
      phi::Place());
  auto abs_op = builder.Build<paddle::dialect::AbsOp>(data_op_x.result(0));

  // y related Ops
  auto data_op_y = builder.Build<paddle::dialect::DataOp>(
      "y",
      std::vector<int64_t>({-1, 128, 32}),
      phi::DataType::FLOAT32,
      phi::Place());
  auto reshape_op = builder.Build<paddle::dialect::ReshapeOp>(
      data_op_y.result(0), std::vector<int64_t>({1, 64, -1, 2}));
  auto cast_op = builder.Build<paddle::dialect::CastOp>(reshape_op.result(0),
                                                        phi::DataType::FLOAT32);

  // z related Ops
  auto data_op_z = builder.Build<paddle::dialect::DataOp>(
      "z",
      std::vector<int64_t>({-1, -1, -1, -1}),
      phi::DataType::FLOAT32,
      phi::Place());
  auto slice_op = builder.Build<paddle::dialect::SliceOp>(
      data_op_z.result(0),
      std::vector<int64_t>({0, 1, 2, 3}),
      std::vector<int64_t>({2, 2, 2, 2}),
      std::vector<int64_t>({-2, -2, -2, -2}),
      std::vector<int64_t>({}),
      std::vector<int64_t>({}));

  auto relu_op = builder.Build<paddle::dialect::ReluOp>(slice_op.result(0));

  // Binary Ops
  auto add_op = builder.Build<paddle::dialect::AddOp>(abs_op.result(0),
                                                      cast_op.result(0));
  auto subtract_op = builder.Build<paddle::dialect::SubtractOp>(
      add_op.result(0), relu_op.result(0));

  // Run shape optimization pass
  pir::PassManager pm(ctx);
  pm.EnableIRPrinting();
  pm.AddPass(pir::CreateShapeOptimizationPass());
  pm.Run(&program);

  // Check the results
  pir::ShapeConstraintIRAnalysis& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(&program);

  symbol::ShapeOrDataDimExprs cast_res =
      shape_analysis.GetShapeOrDataForValue(cast_op.result(0));
  symbol::ShapeOrDataDimExprs relu_res =
      shape_analysis.GetShapeOrDataForValue(relu_op.result(0));
  symbol::ShapeOrDataDimExprs subtract_res =
      shape_analysis.GetShapeOrDataForValue(subtract_op.result(0));

  // TODO(zhangbopd): after shape infer is completed, we can check the results

  // IR_ENFORCE(cast_res.shape()[0] == 1);
  // IR_ENFORCE(cast_res.shape()[1] == 64);
  // IR_ENFORCE(symbol::ToString(cast_res.shape()[2]) == "Mul(S0, 32)");
  // IR_ENFORCE(cast_res.shape()[3] == 2);

  // IR_ENFORCE(symbol::ToString(relu_res.shape()[2]) == "Add(S2, -4)");
  // IR_ENFORCE(symbol::ToString(relu_res.shape()[2]) == "Add(S3, -4)");
  // IR_ENFORCE(symbol::ToString(relu_res.shape()[2]) == "Add(S4, -4)");
  // IR_ENFORCE(symbol::ToString(relu_res.shape()[2]) == "Add(S5, -4)");

  IR_ENFORCE(subtract_res.shape()[0] == 1);
  IR_ENFORCE(subtract_res.shape()[1] == 64);
  IR_ENFORCE(symbol::ToString(subtract_res.shape()[2]) == "Broadcast(S0, -1)");
}
