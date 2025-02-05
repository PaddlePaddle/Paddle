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
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

using cinn::hlir::framework::pir::CompatibleInfo;
using cinn::hlir::framework::pir::OpLoweringGroup;
using cinn::hlir::framework::pir::OpLoweringGroupPtr;

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

std::vector<::pir::Type> CreateDenseTensorTypes(const phi::DDim& dims) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ::pir::Type fp32_dtype = ::pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {};
  size_t offset = 0;
  std::vector<::pir::Type> op_output_types = {::pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset)};
  return op_output_types;
}

std::tuple<std::shared_ptr<::pir::Program>, std::vector<OpLoweringGroupPtr>>
BuildGroupProgramForLowering() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  const std::vector<int64_t> x_shape = {-1, 2};
  const std::vector<int64_t> y_shape = {1, -1, 2};

  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "input_x", x_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto y = builder
               .Build<paddle::dialect::DataOp>(
                   "input_y", y_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim({1, -1, 2})));
  builder.SetInsertionPointToBlockEnd(group_op.block());
  auto exp = builder.Build<paddle::dialect::ExpOp>(x);
  auto reshape = builder.Build<cinn::dialect::ReshapeOp>(
      exp.result(0), std::vector<int>{-1, 1, 1});
  auto sub = builder.Build<paddle::dialect::SubtractOp>(y, reshape.result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{sub.result(0)});

  builder.SetInsertionPointToBlockEnd(program->block());
  builder.Build<paddle::dialect::FetchOp>(group_op->result(0), "out", 0);

  std::vector<OpLoweringGroupPtr> groups;
  groups.emplace_back(std::make_shared<OpLoweringGroup>(
      std::vector<::pir::Operation*>(
          {exp.operation(), reshape.operation(), sub.operation()}),
      CompatibleInfo::GroupOpsName(std::vector<::pir::Operation*>(
          {exp.operation(), reshape.operation(), sub.operation()}))));
  groups[0]->mut_output_ops().insert(groups[0]->ops().back());
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_data;
  symbol::DimExpr x_dim_0("S0");
  symbol::DimExpr x_dim_1(2);
  symbol::DimExpr y_dim_0(1);
  symbol::DimExpr y_dim_1("S1");
  symbol::DimExpr y_dim_2(2);
  value_to_shape_data.emplace(
      x,
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({x_dim_0, x_dim_1})));
  value_to_shape_data.emplace(
      y,
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({y_dim_0, y_dim_1, y_dim_2})));
  value_to_shape_data.emplace(exp.result(0), value_to_shape_data.at(x));
  value_to_shape_data.emplace(reshape.result(0), value_to_shape_data.at(y));
  value_to_shape_data.emplace(sub.result(0), value_to_shape_data.at(y));
  groups[0]->set_value_to_shape_or_data_exprs(value_to_shape_data);

  return {program, groups};
}

std::tuple<std::shared_ptr<::pir::Program>, std::vector<OpLoweringGroupPtr>>
BuildBroadcastGroupProgramForLowering() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  const std::vector<int64_t> x_shape = {1, 1, 1};
  const std::vector<int64_t> y_shape = {1, -1, 128};

  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "input_x", x_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto y = builder
               .Build<paddle::dialect::DataOp>(
                   "input_y", y_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim({1, -1, 128})));
  builder.SetInsertionPointToBlockEnd(group_op.block());
  const std::vector<int64_t> x_broadcast_axes = {0, 1, 2};
  auto x_broadcast =
      builder.Build<cinn::dialect::BroadcastOp>(x, x_broadcast_axes, y_shape);
  auto sub =
      builder.Build<paddle::dialect::SubtractOp>(x_broadcast->result(0), y);
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{sub.result(0)});

  builder.SetInsertionPointToBlockEnd(program->block());
  builder.Build<paddle::dialect::FetchOp>(group_op->result(0), "out", 0);

  std::vector<OpLoweringGroupPtr> groups;
  groups.emplace_back(std::make_shared<OpLoweringGroup>(
      std::vector<::pir::Operation*>(
          {x_broadcast.operation(), sub.operation()}),
      CompatibleInfo::GroupOpsName(std::vector<::pir::Operation*>(
          {x_broadcast.operation(), sub.operation()}))));
  groups[0]->mut_output_ops().insert(groups[0]->ops().back());

  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_data;
  symbol::DimExpr x_dim_0(1);
  symbol::DimExpr x_dim_1(1);
  symbol::DimExpr x_dim_2(1);
  symbol::DimExpr y_dim_0(1);
  symbol::DimExpr y_dim_1("S0");
  symbol::DimExpr y_dim_2(128);
  value_to_shape_data.emplace(
      x,
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({x_dim_0, x_dim_1, x_dim_2})));
  value_to_shape_data.emplace(
      y,
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({y_dim_0, y_dim_1, y_dim_2})));
  value_to_shape_data.emplace(
      x_broadcast.result(0),
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({y_dim_0, y_dim_1, y_dim_2})));
  value_to_shape_data.emplace(
      sub.result(0),
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({y_dim_0, y_dim_1, y_dim_2})));
  groups[0]->set_value_to_shape_or_data_exprs(value_to_shape_data);

  return {program, groups};
}
