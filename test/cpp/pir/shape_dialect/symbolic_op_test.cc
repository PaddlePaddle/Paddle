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

TEST(assist_struct_test, symbolic_dim_mgr_simple) {
  /******************************************************/
  /* Mgr simple version, only SymbolicDim related func. */
  /******************************************************/
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

TEST(assist_struct_test, symbolic_dim_mgr_complex) {
  /***************************************************************/
  /* Mgr with constraintOp, and SymbolicDimProduct related func. */
  /***************************************************************/
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::dialect::SymbolicDim symDimS0 = builder.Build<pir::dialect::SymbolicDim>(
      "S0", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS1 = builder.Build<pir::dialect::SymbolicDim>(
      "S1", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS2 = builder.Build<pir::dialect::SymbolicDim>(
      "S2", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS3 = builder.Build<pir::dialect::SymbolicDim>(
      "S3", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS4 = builder.Build<pir::dialect::SymbolicDim>(
      "S4", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS5 = builder.Build<pir::dialect::SymbolicDim>(
      "S5", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS6 = builder.Build<pir::dialect::SymbolicDim>(
      "S6", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS7 = builder.Build<pir::dialect::SymbolicDim>(
      "S7", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS8 = builder.Build<pir::dialect::SymbolicDim>(
      "S8", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS9 = builder.Build<pir::dialect::SymbolicDim>(
      "S9", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS10 =
      builder.Build<pir::dialect::SymbolicDim>(
          "S10", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS11 =
      builder.Build<pir::dialect::SymbolicDim>(
          "S11", -100000, false, false, true, true);
  pir::dialect::SymbolicDim symDimS12 =
      builder.Build<pir::dialect::SymbolicDim>(
          "S12", -100000, false, false, true, false);
  pir::dialect::SymbolicDim symDimC10 =
      builder.Build<pir::dialect::SymbolicDim>(
          "C10", 10, true, false, true, true);
  pir::dialect::SymbolicDim symDimC20 =
      builder.Build<pir::dialect::SymbolicDim>(
          "C20", 20, true, false, true, true);

  pir::OpResult dimOpS0 = builder.Build<pir::dialect::DimOp>("S0").out();
  pir::OpResult dimOpS1 = builder.Build<pir::dialect::DimOp>("S1").out();
  pir::OpResult dimOpS2 = builder.Build<pir::dialect::DimOp>("S2").out();
  pir::OpResult dimOpS3 = builder.Build<pir::dialect::DimOp>("S3").out();
  pir::OpResult dimOpS4 = builder.Build<pir::dialect::DimOp>("S4").out();
  pir::OpResult dimOpS5 = builder.Build<pir::dialect::DimOp>("S5").out();
  pir::OpResult dimOpS6 = builder.Build<pir::dialect::DimOp>("S6").out();
  pir::OpResult dimOpS7 = builder.Build<pir::dialect::DimOp>("S7").out();
  pir::OpResult dimOpS8 = builder.Build<pir::dialect::DimOp>("S8").out();
  pir::OpResult dimOpS9 = builder.Build<pir::dialect::DimOp>("S9").out();
  pir::OpResult dimOpS10 = builder.Build<pir::dialect::DimOp>("S10").out();
  pir::OpResult dimOpS11 = builder.Build<pir::dialect::DimOp>("S11").out();
  pir::OpResult dimOpC10 = builder.Build<pir::dialect::DimOp>("C10").out();
  pir::OpResult dimOpC20 = builder.Build<pir::dialect::DimOp>("C20").out();
  pir::OpResult constant =
      builder
          .Build<pir::ConstantOp>(pir::Int32Attribute::get(ctx, 2),
                                  pir::Int32Type::get(ctx))
          ->result(0);

  // Mark S1 == S2.
  builder.Build<pir::dialect::TieProductEqualOp>(
      2, 2, std::vector<pir::OpResult>{constant, dimOpS1, dimOpS2, constant});
  // Mark S0 * S1 == S2 * S3, For check S0 == S3.
  builder.Build<pir::dialect::TieProductEqualOp>(
      2, 2, std::vector<pir::OpResult>{dimOpS0, dimOpS1, dimOpS2, dimOpS3});
  // Mark S4 * S0 * S1 == S2 * S3 * S5, For check S4 == S5.
  builder.Build<pir::dialect::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::OpResult>{
          dimOpS4, dimOpS0, dimOpS1, dimOpS2, dimOpS3, dimOpS5});
  // For check S6 == C10 * C20.
  builder.Build<pir::dialect::TieProductEqualOp>(
      1, 2, std::vector<pir::OpResult>{dimOpS6, dimOpC10, dimOpC20});
  // Mark C10 * S0 * S1 == S2 * S3 * S7, for check C10 == S7.
  builder.Build<pir::dialect::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::OpResult>{
          dimOpC10, dimOpS0, dimOpS1, dimOpS2, dimOpS3, dimOpS7});

  // Mark S8 * S9 == S10 * S11, for unsimplify product case
  builder.Build<pir::dialect::TieProductEqualOp>(
      2, 2, std::vector<pir::OpResult>{dimOpS8, dimOpS9, dimOpS10, dimOpS11});

  pir::SymbolicDimMgr symDimMgr(program.module_op());

  symDimMgr.load();

  // For check indirect equality: S1 * S4 == S2 * S5
  pir::SymbolicDimProduct symDimProductLhs;
  pir::SymbolicDimProduct symDimProductRhs;

  symDimProductLhs.symbols.push_back(symDimS1);
  symDimProductLhs.symbols.push_back(symDimS4);

  symDimProductRhs.symbols.push_back(symDimS2);
  symDimProductRhs.symbols.push_back(symDimS5);

  // For uncompletely simplied product check: S8 * S9 * S12 == S10 * S11 * S12
  pir::SymbolicDimProduct symDimProductLhs_;
  pir::SymbolicDimProduct symDimProductRhs_;

  symDimProductLhs_.symbols.push_back(symDimS8);
  symDimProductLhs_.symbols.push_back(symDimS9);
  symDimProductLhs_.symbols.push_back(symDimS12);

  symDimProductRhs_.symbols.push_back(symDimS10);
  symDimProductRhs_.symbols.push_back(symDimS11);
  symDimProductRhs_.symbols.push_back(symDimS12);

  // For check simplifySymbolicDimProduct, {factor = 1, Sym = {S7}} => {factor =
  // 10}
  pir::SymbolicDimProduct symDimProductS7;
  symDimProductS7.symbols.push_back(symDimS7);
  pir::SymbolicDimProduct simplifiedProductS7 =
      symDimMgr.simplifySymbolicDimProduct(symDimProductS7);

  // For check simplifySymbolicDimProductPair, X * Y * Y, Y * Y * Z => X, Z
  pir::SymbolicDimProduct symDimProductPairLhs;
  pir::SymbolicDimProduct symDimProductPairRhs;
  pir::SymbolicDimProduct newLhs, newRhs;
  symDimProductPairLhs.symbols.push_back(symDimS4);
  symDimProductPairLhs.symbols.push_back(symDimS1);
  symDimProductPairLhs.symbols.push_back(symDimS2);
  symDimProductPairRhs.symbols.push_back(symDimS1);
  symDimProductPairRhs.symbols.push_back(symDimS2);
  symDimProductPairRhs.symbols.push_back(symDimS3);

  std::tie(newLhs, newRhs) = symDimMgr.simplifySymbolicDimProductPair(
      symDimProductPairLhs, symDimProductPairRhs);

  // For check symbolicDimProductDivide, {S4 * S1 * C20} / {S1 * C10} => {factor
  // = 2 Sym = {S4}}
  pir::SymbolicDimProduct symDimProductDivLhs;
  pir::SymbolicDimProduct symDimProductDivRhs;
  symDimProductDivLhs.symbols.push_back(symDimS4);
  symDimProductDivLhs.symbols.push_back(symDimS1);
  symDimProductDivLhs.symbols.push_back(symDimC20);
  symDimProductDivRhs.symbols.push_back(symDimS1);
  symDimProductDivRhs.symbols.push_back(symDimC10);

  pir::SymbolicDimProduct *divRes = symDimMgr.symbolicDimProductDivide(
      symDimProductDivLhs, symDimProductDivRhs);

  EXPECT_TRUE(symDimMgr.isSymbolicDimEqual(symDimS1, symDimS2));
  EXPECT_TRUE(symDimMgr.isSymbolicDimEqual(symDimS0, symDimS3));
  EXPECT_TRUE(symDimMgr.isSymbolicDimEqual(symDimS4, symDimS5));
  EXPECT_EQ(symDimS6.getValue(), 200);
  EXPECT_EQ(symDimMgr.symbolTable().lookup<pir::dialect::SymbolicDim>("C20"),
            symDimC20);
  EXPECT_EQ(symDimS7.getValue(), symDimC10.getValue());
  EXPECT_EQ(simplifiedProductS7.factor, 10);
  EXPECT_EQ(simplifiedProductS7.symbols.size(), static_cast<size_t>(0));
  EXPECT_EQ(newLhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(newRhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(newLhs.symbols[0], symDimMgr.getRootSymbolicDim(symDimS4));
  EXPECT_EQ(newRhs.symbols[0], symDimMgr.getRootSymbolicDim(symDimS3));
  EXPECT_EQ(divRes->factor, 2);
  EXPECT_EQ(divRes->symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(divRes->symbols[0], symDimMgr.getRootSymbolicDim(symDimS4));
  EXPECT_TRUE(
      symDimMgr.isSymbolicDimProductEqual(symDimProductLhs, symDimProductRhs));
  EXPECT_TRUE(symDimMgr.isSymbolicDimProductEqual(symDimProductLhs_,
                                                  symDimProductRhs_));
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
