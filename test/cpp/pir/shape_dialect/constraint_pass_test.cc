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
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
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
#include "paddle/pir/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"

pir::AttributeMap CreateAttributeMap(
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::AttributeMap attr_map;
  for (size_t i = 0; i < attribute_names.size(); i++) {
    pir::Attribute attr_value = pir::StrAttribute::get(ctx, attributes[i]);
    attr_map.insert(
        std::pair<std::string, pir::Attribute>(attribute_names[i], attr_value));
  }
  return attr_map;
}

pir::Operation *CreateDenseTensorOp(
    pir::IrContext *ctx,
    const phi::DDim &dims,
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes,
    const pir::Type &dtype =
        pir::Float32Type::get(pir::IrContext::Instance())) {
  std::vector<pir::Value> op_inputs = {};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  std::vector<pir::Type> op_output_types = {
      paddle::dialect::DenseTensorType::get(
          ctx, dtype, dims, data_layout, lod, offset)};
  pir::Operation *op =
      pir::Operation::Create(op_inputs,
                             CreateAttributeMap(attribute_names, attributes),
                             op_output_types,
                             pir::OpInfo());
  return op;
}

TEST(constraint_pass, materialize_and_build_shape) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::PassManager pm(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Operation *op0 = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op0_attr"}, {"op0_name"});
  program.block()->push_back(op0);
  pir::Operation *op1 =
      CreateDenseTensorOp(ctx,
                          {pir::ShapedTypeInterface::kDynamic, 2, 2},
                          {"op1_attr"},
                          {"op1_name"});
  program.block()->push_back(op1);

  EXPECT_EQ(program.block()->size(), static_cast<size_t>(2));
  pm.AddPass(pir::CreateShapeOptimizationPass());

  EXPECT_TRUE(pm.Run(&program));

  // 5 ConstantOp + 5 TensorDim + 2 TieShape + op0 + op1 + 1 funcOp == 15 Ops.
  EXPECT_EQ(program.block()->size(), static_cast<size_t>(15));

  std::stringstream ss;
  program.Print(ss);

  LOG(INFO) << ss.str();
}

TEST(constraint_pass, shape_computation_run) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::PassManager pm(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  builder.Build<pir::dialect::FuncOp>();
  pir::Operation *op0 =
      CreateDenseTensorOp(ctx,
                          {2},
                          {"op0_attr"},
                          {"op0_name"},
                          pir::Int64Type::get(pir::IrContext::Instance()));
  program.block()->push_back(op0);
  pir::Operation *op1 = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op1_attr"}, {"op1_name"});
  program.block()->push_back(op1);

  pm.AddPass(pir::CreateShapeOptimizationPass());

  EXPECT_TRUE(pm.Run(&program));
  pir::SymbolicDimMgr mgr(program.module_op());
  EXPECT_TRUE(mgr.Load());
  pir::ShapeComputationIRAnalysis analysis(program.module_op(), mgr);
  EXPECT_TRUE(analysis.Run());
  EXPECT_FALSE(analysis.Run());
  EXPECT_TRUE(mgr.Save());
}
