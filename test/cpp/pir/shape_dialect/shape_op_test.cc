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

#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include <gtest/gtest.h>
#include <map>
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include "paddle/pir/dialect/shape/utils/symbol_table.h"

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
    const std::vector<std::string> &attributes) {
  std::vector<pir::Value> op_inputs = {};
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  std::vector<pir::Type> op_output_types = {
      paddle::dialect::DenseTensorType::get(
          ctx, fp32_dtype, dims, data_layout, lod, offset)};
  pir::Operation *op =
      pir::Operation::Create(op_inputs,
                             CreateAttributeMap(attribute_names, attributes),
                             op_output_types,
                             pir::OpInfo());
  return op;
}

TEST(shape_op, dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::dialect::DimOp dim_op = builder.Build<pir::dialect::DimOp>("S0");
  pir::OpResult res = dim_op.out();
  EXPECT_EQ(dim_op.getName(), "S0");
  dim_op.setName("S1");
  EXPECT_EQ(dim_op.getName(), "S1");
  EXPECT_EQ(res.owner(), dim_op.operation());
  EXPECT_EQ(res.type(), pir::IndexType::get(ctx));
}

TEST(shape_op, tie_product_equal) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::SymbolTable symbolt_table(program.module_op());

  pir::OpResult dim_op0 = builder.Build<pir::dialect::DimOp>("S0").out();
  pir::OpResult dim_op1 = builder.Build<pir::dialect::DimOp>("S1").out();
  pir::OpResult dim_op2 = builder.Build<pir::dialect::DimOp>("S2").out();
  pir::OpResult dim_op3 = builder.Build<pir::dialect::DimOp>("S3").out();
  pir::OpResult dim_op4 = builder.Build<pir::dialect::DimOp>("S4").out();

  pir::dialect::TieProductEqualOp tie_product_equal =
      builder.Build<pir::dialect::TieProductEqualOp>(
          2,
          3,
          std::vector<pir::Value>{dim_op0, dim_op1, dim_op2, dim_op3, dim_op4});

  std::vector<pir::Value> lhs = tie_product_equal.lhs();
  std::vector<pir::Value> rhs = tie_product_equal.rhs();

  std::vector<pir::Value> lhs_ref{dim_op0, dim_op1};
  std::vector<pir::Value> rhs_ref{dim_op2, dim_op3, dim_op4};

  EXPECT_EQ(symbolt_table.insert(tie_product_equal), "tie_product_equal");
  EXPECT_EQ(
      symbolt_table.Lookup<pir::dialect::TieProductEqualOp>("tie_product_equal")
          .size(),
      static_cast<size_t>(1));
  EXPECT_EQ(symbolt_table.Lookup<pir::dialect::TieProductEqualOp>(
                "tie_product_equal")[0],
            tie_product_equal);
  EXPECT_EQ(lhs, lhs_ref);
  EXPECT_EQ(rhs, rhs_ref);
}

TEST(shape_op, tie_shape) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  auto op = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res = op->result(0);

  pir::dialect::TieShapeOp tie_shape_op =
      builder.Build<pir::dialect::TieShapeOp>(res);
  pir::Value tie_shape_op_value = tie_shape_op.value();

  pir::Attribute attr_s0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attr_s1 = pir::StrAttribute::get(ctx, "S1");

  std::vector<pir::Attribute> new_attrs = {attr_s0, attr_s1};

  auto array_attr = pir::ArrayAttribute::get(ctx, new_attrs);
  tie_shape_op->set_attribute(
      pir::dialect::SymbolicDim::GetSymbolicDimAttrName(), array_attr);

  std::vector<pir::Attribute> arr_attr_vec =
      tie_shape_op
          ->attribute<pir::ArrayAttribute>(
              pir::dialect::SymbolicDim::GetSymbolicDimAttrName())
          .AsVector();

  EXPECT_EQ(tie_shape_op_value, res);
  EXPECT_EQ(arr_attr_vec.size(), static_cast<size_t>(2));
  EXPECT_EQ(arr_attr_vec[0].dyn_cast<pir::StrAttribute>(), attr_s0);
  EXPECT_EQ(arr_attr_vec[1].dyn_cast<pir::StrAttribute>(), attr_s1);
  EXPECT_TRUE(tie_shape_op->HasAttribute(
      pir::dialect::SymbolicDim::GetSymbolicDimAttrName()));
}

TEST(shape_op, func_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::dialect::FuncOp func_op = builder.Build<pir::dialect::FuncOp>();
  auto func_block = func_op.block();
  builder.SetInsertionPointToStart(func_block);
  builder.Build<pir::ConstantOp>(pir::Int32Attribute::get(ctx, 2),
                                 pir::Int32Type::get(ctx));
  EXPECT_EQ(func_block, func_op->region(0).front());
  EXPECT_EQ(func_op->region(0).size(), static_cast<size_t>(1));
  EXPECT_EQ(func_block->size(), static_cast<size_t>(1));
}

TEST(shape_op, tensor_dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::Operation *op = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res_dense_tensor_value = op->result(0);

  pir::dialect::TensorDimOp tensor_dim_op0 =
      builder.Build<pir::dialect::TensorDimOp>(res_dense_tensor_value, 0);
  pir::OpResult res0 = tensor_dim_op0.out();

  pir::OpResult index_value =
      builder
          .Build<pir::ConstantOp>(
              pir::Int64Attribute::get(pir::IrContext::Instance(), 1),
              pir::IndexType::get(pir::IrContext::Instance()))
          ->result(0);
  pir::dialect::TensorDimOp tensor_dim_op1 =
      builder.Build<pir::dialect::TensorDimOp>(res_dense_tensor_value,
                                               index_value);
  pir::OpResult res1 = tensor_dim_op1.out();

  EXPECT_EQ(res0.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(res1.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(tensor_dim_op0.source(), res_dense_tensor_value);
  EXPECT_EQ(tensor_dim_op1.source(), res_dense_tensor_value);
  EXPECT_EQ(tensor_dim_op1.index(), index_value);
}
