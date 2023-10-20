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
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/utils/symbol_table.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(shape_op, symbolic_dim_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::shape::SymbolicDimOp sym_dim_op1 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S0", 10, false, false, false, false);
  pir::shape::SymbolicDimOp sym_dim_op2 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S1", 10, false, false, false, false);

  EXPECT_EQ(sym_dim_op1.GetDimSize(), 10);
  EXPECT_EQ(sym_dim_op1.GetSymName(), "S0");
  EXPECT_FALSE(sym_dim_op1.GetKnownNegativeOne());
  EXPECT_FALSE(sym_dim_op1.GetKnownNonSizeOne());
  EXPECT_FALSE(sym_dim_op1.GetKnownNonSizeZero());
  EXPECT_FALSE(sym_dim_op1.GetKnownNonNegative());

  EXPECT_FALSE(sym_dim_op1.IsDynamic());
  EXPECT_TRUE(sym_dim_op1.Merge(sym_dim_op2));

  sym_dim_op1.SetDimSize(20);
  sym_dim_op1.SetSymName("S2");
  sym_dim_op1.UpdateKnownNegativeOne(true);
  sym_dim_op1.UpdateKnownNonSizeOne(true);
  sym_dim_op1.UpdateKnownNonSizeZero(true);
  sym_dim_op1.UpdateKnownNonNegative(true);

  EXPECT_FALSE(sym_dim_op1.Merge(sym_dim_op2));

  EXPECT_EQ(sym_dim_op1.GetDimSize(), 20);
  EXPECT_EQ(sym_dim_op1.GetSymName(), "S2");
  EXPECT_TRUE(sym_dim_op1.GetKnownNegativeOne());
  EXPECT_TRUE(sym_dim_op1.GetKnownNonSizeOne());
  EXPECT_TRUE(sym_dim_op1.GetKnownNonSizeZero());
  EXPECT_TRUE(sym_dim_op1.GetKnownNonNegative());
}

TEST(shape_op, dim_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::shape::DimOp dim_op = builder.Build<pir::shape::DimOp>("S0");
  pir::OpResult res = dim_op.out();
  EXPECT_EQ(dim_op.GetName(), "S0");
  dim_op.SetName("S1");
  EXPECT_EQ(dim_op.GetName(), "S1");
  EXPECT_EQ(res.owner(), dim_op.operation());
  EXPECT_EQ(res.type(), pir::IndexType::get(ctx));
}

TEST(shape_op, tie_product_equal_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::SymbolTable symbolt_table(program.module_op());

  pir::OpResult dim_op0 = builder.Build<pir::shape::DimOp>("S0").out();
  pir::OpResult dim_op1 = builder.Build<pir::shape::DimOp>("S1").out();
  pir::OpResult dim_op2 = builder.Build<pir::shape::DimOp>("S2").out();
  pir::OpResult dim_op3 = builder.Build<pir::shape::DimOp>("S3").out();
  pir::OpResult dim_op4 = builder.Build<pir::shape::DimOp>("S4").out();

  pir::shape::TieProductEqualOp tie_product_equal_op =
      builder.Build<pir::shape::TieProductEqualOp>(
          2,
          3,
          std::vector<pir::Value>{dim_op0, dim_op1, dim_op2, dim_op3, dim_op4});

  std::vector<pir::Value> lhs = tie_product_equal_op.lhs();
  std::vector<pir::Value> rhs = tie_product_equal_op.rhs();

  std::vector<pir::Value> lhs_ref{dim_op0, dim_op1};
  std::vector<pir::Value> rhs_ref{dim_op2, dim_op3, dim_op4};

  EXPECT_EQ(symbolt_table.insert(tie_product_equal_op), "tie_product_equal");
  EXPECT_EQ(
      symbolt_table.Lookup<pir::shape::TieProductEqualOp>("tie_product_equal")
          .size(),
      static_cast<size_t>(1));
  EXPECT_EQ(symbolt_table.Lookup<pir::shape::TieProductEqualOp>(
                "tie_product_equal")[0],
            tie_product_equal_op);
  EXPECT_EQ(lhs, lhs_ref);
  EXPECT_EQ(rhs, rhs_ref);
}

TEST(shape_op, tie_shape_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  auto op = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res = op->result(0);

  pir::shape::TieShapeOp tie_shape_op =
      builder.Build<pir::shape::TieShapeOp>(res);
  pir::Value tie_shape_op_input = tie_shape_op.input();

  pir::Attribute attr_s0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attr_s1 = pir::StrAttribute::get(ctx, "S1");

  std::vector<pir::Attribute> new_attrs = {attr_s0, attr_s1};

  auto array_attr = pir::ArrayAttribute::get(ctx, new_attrs);
  tie_shape_op->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), array_attr);

  std::vector<pir::Attribute> arr_attr_vec =
      tie_shape_op
          ->attribute<pir::ArrayAttribute>(
              pir::shape::SymbolicDimOp::GetSymbolicDimAttrName())
          .AsVector();

  EXPECT_EQ(tie_shape_op_input, res);
  EXPECT_EQ(arr_attr_vec.size(), static_cast<size_t>(2));
  EXPECT_EQ(arr_attr_vec[0].dyn_cast<pir::StrAttribute>(), attr_s0);
  EXPECT_EQ(arr_attr_vec[1].dyn_cast<pir::StrAttribute>(), attr_s1);
  EXPECT_TRUE(tie_shape_op->HasAttribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName()));
}

TEST(shape_op, func_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::shape::FuncOp func_op = builder.Build<pir::shape::FuncOp>();
  auto func_block = func_op.block();
  builder.SetInsertionPointToStart(func_block);
  builder.Build<pir::ConstantOp>(pir::Int32Attribute::get(ctx, 2),
                                 pir::Int32Type::get(ctx));
  EXPECT_EQ(func_block, func_op->region(0).front());
  EXPECT_EQ(func_op->region(0).size(), static_cast<size_t>(1));
  EXPECT_EQ(func_block->size(), static_cast<size_t>(1));
}

TEST(shape_op, tensor_dim_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::Operation *op = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res_dense_tensor_value = op->result(0);

  pir::shape::TensorDimOp tensor_dim_op0 =
      builder.Build<pir::shape::TensorDimOp>(res_dense_tensor_value, 0);
  pir::OpResult res0 = tensor_dim_op0.out();

  pir::OpResult index_value =
      builder
          .Build<pir::ConstantOp>(
              pir::Int64Attribute::get(pir::IrContext::Instance(), 1),
              pir::IndexType::get(pir::IrContext::Instance()))
          ->result(0);
  pir::shape::TensorDimOp tensor_dim_op1 =
      builder.Build<pir::shape::TensorDimOp>(res_dense_tensor_value,
                                             index_value);
  pir::OpResult res1 = tensor_dim_op1.out();

  EXPECT_EQ(res0.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(res1.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(tensor_dim_op0.source(), res_dense_tensor_value);
  EXPECT_EQ(tensor_dim_op1.source(), res_dense_tensor_value);
  EXPECT_EQ(tensor_dim_op1.index(), index_value);
}

TEST(shape_op, shape_of_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  auto op = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res = op->result(0);

  pir::shape::ShapeOfOp shape_of_op = builder.Build<pir::shape::ShapeOfOp>(res);
  pir::Value shape_of_op_input = shape_of_op.input();
  EXPECT_EQ(shape_of_op_input, res);
}

TEST(shape_op, from_elements_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  auto op = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult res = op->result(0);

  pir::shape::ShapeOfOp from_elements_op =
      builder.Build<pir::shape::ShapeOfOp>(res);
  pir::Value shape_of_op_input = from_elements_op.input();
  EXPECT_EQ(shape_of_op_input, res);
}

// TODO(zhangbopd): Add more shap_op tests.
