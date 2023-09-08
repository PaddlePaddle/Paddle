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
#include <map>
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"

TEST(assist_struct_test, symbolic_dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::dialect::SymbolicDim symDim = builder.Build<pir::dialect::SymbolicDim>(
      "S0", 10, false, false, false, false);
  pir::dialect::SymbolicDim symDim_ = builder.Build<pir::dialect::SymbolicDim>(
      "S1", 10, false, false, false, false);
  EXPECT_EQ(symDim.getValue(), 10);
  EXPECT_EQ(symDim.getSymName(), "S0");
  EXPECT_FALSE(symDim.getKnownNegativeOne());
  EXPECT_FALSE(symDim.getKnownNonSizeOne());
  EXPECT_FALSE(symDim.getKnownNonSizeZero());
  EXPECT_FALSE(symDim.getKnownNonNegative());

  EXPECT_FALSE(symDim.isDynamic());
  EXPECT_TRUE(symDim.merge(symDim_));

  symDim.updateValue(20);
  symDim.updateSymName("S2");
  symDim.updateKnownNegativeOne(true);
  symDim.updateKnownNonSizeOne(true);
  symDim.updateKnownNonSizeZero(true);
  symDim.updateKnownNonNegative(true);

  EXPECT_FALSE(symDim.merge(symDim_));

  EXPECT_EQ(symDim.getValue(), 20);
  EXPECT_EQ(symDim.getSymName(), "S2");
  EXPECT_TRUE(symDim.getKnownNegativeOne());
  EXPECT_TRUE(symDim.getKnownNonSizeOne());
  EXPECT_TRUE(symDim.getKnownNonSizeZero());
  EXPECT_TRUE(symDim.getKnownNonNegative());
}

TEST(assist_struct_test, symbolic_dim_product) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::dialect::SymbolicDim symDim = builder.Build<pir::dialect::SymbolicDim>(
      "S0", -100000, false, false, false, false);
  pir::SymbolicDimProduct symDimProduct;
  pir::SymbolicDimProduct symDimProduct_;
  symDimProduct.symbols.push_back(symDim);
  symDimProduct.factor *= 10;
  EXPECT_EQ(symDimProduct.factor, 10);
  EXPECT_NE(symDimProduct, symDimProduct_);
  EXPECT_FALSE(symDimProduct.empty());
}

TEST(assist_struct_test, symbolic_dim_table) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::dialect::SymbolicDim symDim = builder.Build<pir::dialect::SymbolicDim>(
      "S0", 10, false, false, false, false);

  pir::SymbolTable symbolTable(program.module_op());
  EXPECT_EQ(symbolTable.insert(symDim), "S0");
  EXPECT_EQ(symbolTable.lookup<pir::dialect::SymbolicDim>("S0"), symDim);
  EXPECT_EQ(symbolTable.getOp(), program.module_op());
  EXPECT_FALSE(symbolTable.lookup<pir::dialect::SymbolicDim>("S1"));
}

TEST(assist_struct_test, symbolic_dim_mgr) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  pir::SymbolicDimMgr symDimMgr(program.module_op());
  pir::dialect::SymbolicDim symDimS0 = symDimMgr.newSymbolicDim();
  pir::dialect::SymbolicDim symDimS1 = symDimMgr.newSymbolicDim();
  pir::dialect::SymbolicDim symDimC10 = symDimMgr.newConstantSymbolicDim(10);
  symDimMgr.mapSymbolicDimEqual(symDimS0, symDimS1);

  pir::Attribute attr_value = pir::StrAttribute::get(ctx, "op_attr");
  pir::AttributeMap attr_map;
  attr_map.insert(std::pair<std::string, pir::Attribute>("op", attr_value));
  std::vector<pir::OpResult> op_inputs = {};

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {-100000, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  std::vector<pir::Type> op_output_types = {
      paddle::dialect::DenseTensorType::get(
          ctx, fp32_dtype, dims, data_layout, lod, offset)};
  pir::Operation *op = pir::Operation::Create(
      op_inputs, attr_map, op_output_types, pir::OpInfo());
  pir::Value res = op->result(0);

  std::vector<pir::dialect::SymbolicDim> symDimVec =
      symDimMgr.createSymbolicDimsForRankedValue(res);

  EXPECT_EQ(symDimS0.getSymName(), "S0");
  EXPECT_EQ(symDimS1.getSymName(), "S1");
  EXPECT_EQ(symDimS1.getValue(), -100000);
  EXPECT_EQ(symDimC10.getSymName(), "C10");
  EXPECT_EQ(symDimC10.getValue(), 10);
  EXPECT_EQ(symDimVec[0].getSymName(), "S2");
  EXPECT_EQ(symDimVec[1].getSymName(), "C2");
  EXPECT_EQ(symDimMgr.symbolTable().lookup<pir::dialect::SymbolicDim>("S0"),
            symDimS0);
  EXPECT_EQ(symDimMgr.symbolTable().lookup<pir::dialect::SymbolicDim>("C10"),
            symDimC10);
  EXPECT_EQ(symDimMgr.getRootSymbolicDim(symDimS1), symDimS0);
  EXPECT_TRUE(symDimMgr.isSymbolicDimEqual(symDimS0, symDimS1));
  EXPECT_FALSE(symDimMgr.isSymbolicDimEqual(symDimS0, symDimC10));
}

TEST(assist_struct_test, dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::dialect::DimOp dimOp = builder.Build<pir::dialect::DimOp>("S0");
  pir::OpResult res = dimOp.out();
  EXPECT_EQ(dimOp.getName(), "S0");
  dimOp.setName("S1");
  EXPECT_EQ(dimOp.getName(), "S1");
  EXPECT_EQ(res.GetDefiningOp(), dimOp.operation());
  EXPECT_EQ(res.type(), pir::IndexType::get(ctx));
}

TEST(assist_struct_test, tie_product_equal) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::SymbolTable symbolTable(program.module_op());

  pir::OpResult dimOp0 = builder.Build<pir::dialect::DimOp>("S0").out();
  pir::OpResult dimOp1 = builder.Build<pir::dialect::DimOp>("S1").out();
  pir::OpResult dimOp2 = builder.Build<pir::dialect::DimOp>("S2").out();
  pir::OpResult dimOp3 = builder.Build<pir::dialect::DimOp>("S3").out();
  pir::OpResult dimOp4 = builder.Build<pir::dialect::DimOp>("S4").out();

  pir::dialect::TieProductEqualOp tie_product_equal =
      builder.Build<pir::dialect::TieProductEqualOp>(
          2,
          3,
          std::vector<pir::OpResult>{dimOp0, dimOp1, dimOp2, dimOp3, dimOp4});

  std::vector<pir::Value> lhs = tie_product_equal.getLhs();
  std::vector<pir::Value> rhs = tie_product_equal.getRhs();

  std::vector<pir::Value> lhs_ref{dimOp0, dimOp1};
  std::vector<pir::Value> rhs_ref{dimOp2, dimOp3, dimOp4};

  EXPECT_EQ(symbolTable.insert(tie_product_equal), "tie_product_equal");
  EXPECT_EQ(
      symbolTable.lookup<pir::dialect::TieProductEqualOp>("tie_product_equal")
          .size(),
      static_cast<size_t>(1));
  EXPECT_EQ(symbolTable.lookup<pir::dialect::TieProductEqualOp>(
                "tie_product_equal")[0],
            tie_product_equal);
  EXPECT_EQ(lhs, lhs_ref);
  EXPECT_EQ(rhs, rhs_ref);
}
