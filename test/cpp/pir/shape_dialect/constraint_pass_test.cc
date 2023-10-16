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
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/cast_utils.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/op_info.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/dialect/shape/transforms/passes.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"

#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(shape_constraint_pass, materialize_and_build_shape) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Operation *op0 =
      test::CreateDenseTensorOp(ctx,
                                {pir::ShapedTypeInterface::kDynamic, 2},
                                {"op0_attr"},
                                {"create_dense_tensor_op0"});
  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx,
                                {pir::ShapedTypeInterface::kDynamic, 2, 2},
                                {"op1_attr"},
                                {"create_dense_tensor_op1"});
  program.block()->push_back(op0);
  program.block()->push_back(op1);

  EXPECT_EQ(program.block()->size(), 2u);

  std::stringstream ss1;
  program.Print(ss1);
  LOG(INFO) << " ================================================  Before Add "
               "and Run Pass ================================================ ";
  LOG(INFO) << ss1.str();

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateShapeOptimizationPass());

  EXPECT_TRUE(pm.Run(&program));

  // 5 ConstantOp + 5 TensorDim + 2 TieShape + op0 + op1 + 1 funcOp == 15 Ops.
  EXPECT_EQ(program.block()->size(), 15u);

  std::stringstream ss2;
  program.Print(ss2);
  LOG(INFO) << " ================================================  After Add "
               "and Run Pass ================================================ ";
  LOG(INFO) << ss2.str();
}

TEST(shape_constraint_pass, shape_computation_run) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Builder builder = ::pir::Builder(ctx, program.block());
  builder.Build<pir::shape::FuncOp>();
  pir::Operation *op0 = test::CreateDenseTensorOp(
      ctx,
      {2},
      {"op0_attr"},
      {"op0_name"},
      pir::Int64Type::get(pir::IrContext::Instance()));
  program.block()->push_back(op0);
  pir::Operation *op1 = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op1_attr"}, {"op1_name"});
  program.block()->push_back(op1);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateShapeOptimizationPass());

  EXPECT_TRUE(pm.Run(&program));
  pir::SymbolicDimMgr mgr(program.module_op());
  EXPECT_TRUE(mgr.Load());
  EXPECT_TRUE(mgr.Save());
}

// TODO(zhangbopd): ExpandShapeOfOpPattern etc.
