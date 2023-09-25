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
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/dialect/shape/utils/shape_utils.h"

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

  EXPECT_FALSE(symDim.IsDynamic());
  EXPECT_TRUE(symDim.Merge(symDim_));

  symDim.updateValue(20);
  symDim.updateSymName("S2");
  symDim.updateKnownNegativeOne(true);
  symDim.updateKnownNonSizeOne(true);
  symDim.updateKnownNonSizeZero(true);
  symDim.updateKnownNonNegative(true);

  EXPECT_FALSE(symDim.Merge(symDim_));

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
      "S0", pir::ShapedTypeInterface::kDynamic, false, false, false, false);
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
  EXPECT_EQ(symbolTable.Lookup<pir::dialect::SymbolicDim>("S0"), symDim);
  EXPECT_EQ(symbolTable.getOp(), program.module_op());
  EXPECT_FALSE(symbolTable.Lookup<pir::dialect::SymbolicDim>("S1"));
}

TEST(assist_struct_test, symbolic_dim_mgr_simple) {
  /******************************************************/
  /* Mgr simple version, only SymbolicDim related func. */
  /******************************************************/
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::SymbolicDimMgr symDimMgr(program.module_op());
  pir::dialect::SymbolicDim symDimS0 = symDimMgr.NewSymbolicDim();
  pir::dialect::SymbolicDim symDimS1 = symDimMgr.NewSymbolicDim();
  pir::dialect::SymbolicDim symDimC10 = symDimMgr.NewConstantSymbolicDim(10);
  symDimMgr.MapSymbolicDimEqual(symDimS0, symDimS1);

  auto op = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::Value res = op->result(0);

  std::vector<pir::dialect::SymbolicDim> symDimVec =
      symDimMgr.CreateSymbolicDimsForRankedValue(res);

  EXPECT_EQ(symDimS0.getSymName(), "S0");
  EXPECT_EQ(symDimS1.getSymName(), "S1");
  EXPECT_EQ(symDimS1.getValue(), pir::ShapedTypeInterface::kDynamic);
  EXPECT_EQ(symDimC10.getSymName(), "C10");
  EXPECT_EQ(symDimC10.getValue(), 10);
  EXPECT_EQ(symDimVec[0].getSymName(), "S2");
  EXPECT_EQ(symDimVec[1].getSymName(), "C2");
  EXPECT_EQ(symDimMgr.symbolTable().Lookup<pir::dialect::SymbolicDim>("S0"),
            symDimS0);
  EXPECT_EQ(symDimMgr.symbolTable().Lookup<pir::dialect::SymbolicDim>("C10"),
            symDimC10);
  EXPECT_EQ(symDimMgr.GetRootSymbolicDim(symDimS1), symDimS0);
  EXPECT_TRUE(symDimMgr.IsSymbolicDimEqual(symDimS0, symDimS1));
  EXPECT_FALSE(symDimMgr.IsSymbolicDimEqual(symDimS0, symDimC10));
}

TEST(assist_struct_test, symbolic_dim_mgr_complex) {
  /***************************************************************/
  /* Mgr with constraintOp, and SymbolicDimProduct related func. */
  /***************************************************************/
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::SymbolicDimMgr symDimMgr(program.module_op());
  auto funcOp =
      symDimMgr.symbolTable().getOp()->dyn_cast<pir::dialect::FuncOp>();

  pir::Builder builder = pir::Builder(ctx, funcOp.block());

  pir::dialect::SymbolicDim symDimS0 = symDimMgr.NewSymbolicDim("S0");
  pir::dialect::SymbolicDim symDimS1 = symDimMgr.NewSymbolicDim("S1");
  pir::dialect::SymbolicDim symDimS2 = symDimMgr.NewSymbolicDim("S2");
  pir::dialect::SymbolicDim symDimS3 = symDimMgr.NewSymbolicDim("S3");
  pir::dialect::SymbolicDim symDimS4 = symDimMgr.NewSymbolicDim("S4");
  pir::dialect::SymbolicDim symDimS5 = symDimMgr.NewSymbolicDim("S5");
  pir::dialect::SymbolicDim symDimS6 = symDimMgr.NewSymbolicDim("S6");
  pir::dialect::SymbolicDim symDimS7 = symDimMgr.NewSymbolicDim("S7");
  pir::dialect::SymbolicDim symDimS8 = symDimMgr.NewSymbolicDim("S8");
  pir::dialect::SymbolicDim symDimS9 = symDimMgr.NewSymbolicDim("S9");
  pir::dialect::SymbolicDim symDimS10 = symDimMgr.NewSymbolicDim("S10");
  pir::dialect::SymbolicDim symDimS11 = symDimMgr.NewSymbolicDim("S11");
  pir::dialect::SymbolicDim symDimS12 = symDimMgr.NewSymbolicDim("S12");
  pir::dialect::SymbolicDim symDimC10 = symDimMgr.NewConstantSymbolicDim(10);
  pir::dialect::SymbolicDim symDimC20 = symDimMgr.NewConstantSymbolicDim(20);

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
      2, 2, std::vector<pir::Value>{constant, dimOpS1, dimOpS2, constant});
  // Mark S0 * S1 == S2 * S3, For check S0 == S3.
  builder.Build<pir::dialect::TieProductEqualOp>(
      2, 2, std::vector<pir::Value>{dimOpS0, dimOpS1, dimOpS2, dimOpS3});
  // Mark S4 * S0 * S1 == S2 * S3 * S5, For check S4 == S5.
  builder.Build<pir::dialect::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::Value>{
          dimOpS4, dimOpS0, dimOpS1, dimOpS2, dimOpS3, dimOpS5});
  // For check S6 == C10 * C20.
  builder.Build<pir::dialect::TieProductEqualOp>(
      1, 2, std::vector<pir::Value>{dimOpS6, dimOpC10, dimOpC20});
  // Mark C10 * S0 * S1 == S2 * S3 * S7, for check C10 == S7.
  builder.Build<pir::dialect::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::Value>{
          dimOpC10, dimOpS0, dimOpS1, dimOpS2, dimOpS3, dimOpS7});

  // For unsimplify product case: S8 * S9 == S10 * S11
  builder.Build<pir::dialect::TieProductEqualOp>(
      2, 2, std::vector<pir::Value>{dimOpS8, dimOpS9, dimOpS10, dimOpS11});

  auto op = CreateDenseTensorOp(ctx,
                                {pir::ShapedTypeInterface::kDynamic,
                                 pir::ShapedTypeInterface::kDynamic,
                                 pir::ShapedTypeInterface::kDynamic,
                                 pir::ShapedTypeInterface::kDynamic,
                                 pir::ShapedTypeInterface::kDynamic,
                                 pir::ShapedTypeInterface::kDynamic},
                                {"op0_attr"},
                                {"op0_name"});
  auto op_ = CreateDenseTensorOp(ctx,
                                 {pir::ShapedTypeInterface::kDynamic,
                                  pir::ShapedTypeInterface::kDynamic,
                                  pir::ShapedTypeInterface::kDynamic,
                                  pir::ShapedTypeInterface::kDynamic,
                                  pir::ShapedTypeInterface::kDynamic,
                                  10,
                                  20},
                                 {"op1_attr"},
                                 {"op1_name"});
  pir::OpResult res = op->result(0);
  pir::OpResult res_ = op_->result(0);

  builder.SetInsertionPointToEnd(program.block());
  pir::dialect::TieShapeOp tieShapeOp =
      builder.Build<pir::dialect::TieShapeOp>(res);
  pir::dialect::TieShapeOp tieShapeOp_ =
      builder.Build<pir::dialect::TieShapeOp>(res_);

  pir::Attribute attrS0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attrS1 = pir::StrAttribute::get(ctx, "S1");
  pir::Attribute attrS2 = pir::StrAttribute::get(ctx, "S2");
  pir::Attribute attrS3 = pir::StrAttribute::get(ctx, "S3");
  pir::Attribute attrS4 = pir::StrAttribute::get(ctx, "S4");
  pir::Attribute attrS5 = pir::StrAttribute::get(ctx, "S5");
  pir::Attribute attrS6 = pir::StrAttribute::get(ctx, "S6");
  pir::Attribute attrS7 = pir::StrAttribute::get(ctx, "S7");
  pir::Attribute attrS8 = pir::StrAttribute::get(ctx, "S8");
  pir::Attribute attrS9 = pir::StrAttribute::get(ctx, "S9");
  pir::Attribute attrS10 = pir::StrAttribute::get(ctx, "S10");
  pir::Attribute attrS11 = pir::StrAttribute::get(ctx, "S11");
  pir::Attribute attrC10 = pir::StrAttribute::get(ctx, "C10");
  pir::Attribute attrC20 = pir::StrAttribute::get(ctx, "C20");

  std::vector<pir::Attribute> newAttrs = {
      attrS0, attrS1, attrS2, attrS3, attrS4, attrS5};
  std::vector<pir::Attribute> newAttrsRef = {
      attrS0, attrS1, attrS1, attrS0, attrS2, attrS2};
  std::vector<pir::Attribute> newAttrs_ = {
      attrS6, attrS7, attrS8, attrS9, attrS10, attrS11, attrC10, attrC20};

  auto arrayAttr = pir::ArrayAttribute::get(ctx, newAttrs);
  auto arrayAttrRef = pir::ArrayAttribute::get(ctx, newAttrsRef);
  auto arrayAttr_ = pir::ArrayAttribute::get(ctx, newAttrs_);
  tieShapeOp->set_attribute(pir::dialect::SymbolicDim::getSymbolicDimAttrName(),
                            arrayAttr);
  tieShapeOp_->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), arrayAttr_);

  EXPECT_TRUE(symDimMgr.Load());

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

  // For check SimplifySymbolicDimProduct, {factor = 1, Sym = {S7}} => {factor =
  // 10}
  pir::SymbolicDimProduct symDimProductS7;
  symDimProductS7.symbols.push_back(symDimS7);
  pir::SymbolicDimProduct simplifiedProductS7 =
      symDimMgr.SimplifySymbolicDimProduct(symDimProductS7);

  // For check SimplifySymbolicDimProductPair, X * Y * Y, Y * Y * Z => X, Z
  pir::SymbolicDimProduct symDimProductPairLhs;
  pir::SymbolicDimProduct symDimProductPairRhs;
  pir::SymbolicDimProduct newLhs, newRhs;
  symDimProductPairLhs.symbols.push_back(symDimS4);
  symDimProductPairLhs.symbols.push_back(symDimS1);
  symDimProductPairLhs.symbols.push_back(symDimS2);
  symDimProductPairRhs.symbols.push_back(symDimS1);
  symDimProductPairRhs.symbols.push_back(symDimS2);
  symDimProductPairRhs.symbols.push_back(symDimS3);

  std::tie(newLhs, newRhs) = symDimMgr.SimplifySymbolicDimProductPair(
      symDimProductPairLhs, symDimProductPairRhs);

  // For check SymbolicDimProductDivide, {S4 * S1 * C20} / {S1 * C10} => {factor
  // = 2 Sym = {S4}}
  pir::SymbolicDimProduct symDimProductDivLhs;
  pir::SymbolicDimProduct symDimProductDivRhs;
  symDimProductDivLhs.symbols.push_back(symDimS4);
  symDimProductDivLhs.symbols.push_back(symDimS1);
  symDimProductDivLhs.symbols.push_back(symDimC20);
  symDimProductDivRhs.symbols.push_back(symDimS1);
  symDimProductDivRhs.symbols.push_back(symDimC10);

  pir::SymbolicDimProduct *divRes = symDimMgr.SymbolicDimProductDivide(
      symDimProductDivLhs, symDimProductDivRhs);

  EXPECT_TRUE(symDimMgr.IsSymbolicDimEqual(symDimS1, symDimS2));
  EXPECT_TRUE(symDimMgr.IsSymbolicDimEqual(symDimS0, symDimS3));
  EXPECT_TRUE(symDimMgr.IsSymbolicDimEqual(symDimS4, symDimS5));
  EXPECT_EQ(symDimS6.getValue(), 200);
  EXPECT_EQ(symDimMgr.symbolTable().Lookup<pir::dialect::SymbolicDim>("C20"),
            symDimC20);
  EXPECT_EQ(symDimS7.getValue(), symDimC10.getValue());
  EXPECT_EQ(simplifiedProductS7.factor, 10);
  EXPECT_EQ(simplifiedProductS7.symbols.size(), static_cast<size_t>(0));
  EXPECT_EQ(newLhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(newRhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(newLhs.symbols[0], symDimMgr.GetRootSymbolicDim(symDimS4));
  EXPECT_EQ(newRhs.symbols[0], symDimMgr.GetRootSymbolicDim(symDimS3));
  EXPECT_EQ(divRes->factor, 2);
  EXPECT_EQ(divRes->symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(divRes->symbols[0], symDimMgr.GetRootSymbolicDim(symDimS4));
  EXPECT_TRUE(
      symDimMgr.IsSymbolicDimProductEqual(symDimProductLhs, symDimProductRhs));
  EXPECT_TRUE(symDimMgr.IsSymbolicDimProductEqual(symDimProductLhs_,
                                                  symDimProductRhs_));
  EXPECT_TRUE(symDimMgr.Save());

  pir::SymbolicDimMgr symDimMgr_(program.module_op());
  EXPECT_TRUE(symDimMgr_.Load());
  auto attrs = tieShapeOp.attribute<pir::ArrayAttribute>(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName());
  EXPECT_FALSE(
      symDimMgr_.symbolTable().Lookup<pir::dialect::SymbolicDim>("S7"));
  EXPECT_EQ(symDimMgr_.symbolTable()
                .Lookup<pir::dialect::TieProductEqualOp>("tie_product_equal")
                .size(),
            static_cast<size_t>(1));

  EXPECT_EQ(attrs.AsVector(), arrayAttrRef.AsVector());
}

TEST(shape_op, dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::dialect::DimOp dimOp = builder.Build<pir::dialect::DimOp>("S0");
  pir::OpResult res = dimOp.out();
  EXPECT_EQ(dimOp.getName(), "S0");
  dimOp.setName("S1");
  EXPECT_EQ(dimOp.getName(), "S1");
  EXPECT_EQ(res.owner(), dimOp.operation());
  EXPECT_EQ(res.type(), pir::IndexType::get(ctx));
}

TEST(shape_op, tie_product_equal) {
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
          std::vector<pir::Value>{dimOp0, dimOp1, dimOp2, dimOp3, dimOp4});

  std::vector<pir::Value> lhs = tie_product_equal.lhs();
  std::vector<pir::Value> rhs = tie_product_equal.rhs();

  std::vector<pir::Value> lhs_ref{dimOp0, dimOp1};
  std::vector<pir::Value> rhs_ref{dimOp2, dimOp3, dimOp4};

  EXPECT_EQ(symbolTable.insert(tie_product_equal), "tie_product_equal");
  EXPECT_EQ(
      symbolTable.Lookup<pir::dialect::TieProductEqualOp>("tie_product_equal")
          .size(),
      static_cast<size_t>(1));
  EXPECT_EQ(symbolTable.Lookup<pir::dialect::TieProductEqualOp>(
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

  pir::dialect::TieShapeOp tieShapeOp =
      builder.Build<pir::dialect::TieShapeOp>(res);
  pir::Value tieShapeOpValue = tieShapeOp.value();

  pir::Attribute attrS0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attrS1 = pir::StrAttribute::get(ctx, "S1");

  std::vector<pir::Attribute> newAttrs = {attrS0, attrS1};

  auto arrayAttr = pir::ArrayAttribute::get(ctx, newAttrs);
  tieShapeOp->set_attribute(pir::dialect::SymbolicDim::getSymbolicDimAttrName(),
                            arrayAttr);

  std::vector<pir::Attribute> arrAttrVec =
      tieShapeOp
          ->attribute<pir::ArrayAttribute>(
              pir::dialect::SymbolicDim::getSymbolicDimAttrName())
          .AsVector();

  EXPECT_EQ(tieShapeOpValue, res);
  EXPECT_EQ(arrAttrVec.size(), static_cast<size_t>(2));
  EXPECT_EQ(arrAttrVec[0].dyn_cast<pir::StrAttribute>(), attrS0);
  EXPECT_EQ(arrAttrVec[1].dyn_cast<pir::StrAttribute>(), attrS1);
  EXPECT_TRUE(tieShapeOp->HasAttribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName()));
}

TEST(shape_op, func_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::dialect::FuncOp funcOp = builder.Build<pir::dialect::FuncOp>();
  auto funcBlock = funcOp.block();
  builder.SetInsertionPointToStart(funcBlock);
  builder.Build<pir::ConstantOp>(pir::Int32Attribute::get(ctx, 2),
                                 pir::Int32Type::get(ctx));
  EXPECT_EQ(funcBlock, funcOp->region(0).front());
  EXPECT_EQ(funcOp->region(0).size(), static_cast<size_t>(1));
  EXPECT_EQ(funcBlock->size(), static_cast<size_t>(1));
}

TEST(assist_struct_test, shape_analysis) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::dialect::FuncOp funcOp = builder.Build<pir::dialect::FuncOp>();

  phi::DDim dims_D_2 = {pir::ShapedTypeInterface::kDynamic, 2};
  phi::DDim dims_2_2 = {2, 2};
  phi::DDim dims_D = {pir::ShapedTypeInterface::kDynamic};

  // same shape with dynamic: value1 == value2
  auto op1 = CreateDenseTensorOp(ctx, dims_D_2, {"op1_attr"}, {"op1_name"});
  auto op2 = CreateDenseTensorOp(ctx, dims_D_2, {"op2_attr"}, {"op2_name"});
  pir::OpResult value1 = op1->result(0);
  pir::OpResult value2 = op2->result(0);

  // same shape with static: value3 == value4
  auto op3 = CreateDenseTensorOp(ctx, dims_2_2, {"op3_attr"}, {"op3_name"});
  auto op4 = CreateDenseTensorOp(ctx, dims_2_2, {"op4_attr"}, {"op4_name"});
  pir::OpResult value3 = op3->result(0);
  pir::OpResult value4 = op4->result(0);

  // one dimension with dynamic: value5 != value1 != value3
  auto op5 = CreateDenseTensorOp(ctx, dims_D, {"op5_attr"}, {"op5_name"});
  pir::OpResult value5 = op5->result(0);

  pir::dialect::TieShapeOp tieShapeOp1 =
      builder.Build<pir::dialect::TieShapeOp>(value1);
  pir::dialect::TieShapeOp tieShapeOp2 =
      builder.Build<pir::dialect::TieShapeOp>(value2);
  pir::dialect::TieShapeOp tieShapeOp3 =
      builder.Build<pir::dialect::TieShapeOp>(value3);
  pir::dialect::TieShapeOp tieShapeOp4 =
      builder.Build<pir::dialect::TieShapeOp>(value4);
  pir::dialect::TieShapeOp tieShapeOp5 =
      builder.Build<pir::dialect::TieShapeOp>(value5);

  builder.SetInsertionPointToEnd(funcOp.block());
  builder.Build<pir::dialect::SymbolicDim>("C2", 2, true, false, true, true);
  pir::dialect::SymbolicDim symDimS0 = builder.Build<pir::dialect::SymbolicDim>(
      "S0", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::dialect::SymbolicDim symDimS1 = builder.Build<pir::dialect::SymbolicDim>(
      "S1", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::dialect::SymbolicDim symDimS2 = builder.Build<pir::dialect::SymbolicDim>(
      "S2", pir::ShapedTypeInterface::kDynamic, false, false, true, true);

  pir::Attribute attrS0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attrS1 = pir::StrAttribute::get(ctx, "S1");
  pir::Attribute attrS2 = pir::StrAttribute::get(ctx, "S2");
  pir::Attribute attrC2 = pir::StrAttribute::get(ctx, "C2");

  auto attrOp1 = pir::ArrayAttribute::get(ctx, {attrS0, attrC2});
  auto attrOp2 = pir::ArrayAttribute::get(ctx, {attrS1, attrC2});
  auto attrOp3 = pir::ArrayAttribute::get(ctx, {attrC2, attrC2});
  auto attrOp4 = pir::ArrayAttribute::get(ctx, {attrC2, attrC2});
  auto attrOp5 = pir::ArrayAttribute::get(ctx, {attrS2});

  tieShapeOp1->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), attrOp1);
  tieShapeOp2->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), attrOp2);
  tieShapeOp3->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), attrOp3);
  tieShapeOp4->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), attrOp4);
  tieShapeOp5->set_attribute(
      pir::dialect::SymbolicDim::getSymbolicDimAttrName(), attrOp5);

  pir::SymbolicDimShapeAnalysis shapeAnalysis(program.module_op());
  EXPECT_TRUE(shapeAnalysis.IsShapeEqual(value3, value4));
  EXPECT_FALSE(shapeAnalysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shapeAnalysis.IsShapeEqual(value1, value3));
  EXPECT_FALSE(shapeAnalysis.IsShapeEqual(value1, value5));
  EXPECT_FALSE(shapeAnalysis.IsShapeEqual(value3, value5));
  EXPECT_TRUE(shapeAnalysis.IsProductEqual(value1, {1}, value3, {0}));
  EXPECT_TRUE(shapeAnalysis.IsSameNumElements(value4, value3));

  shapeAnalysis.symbolicDimMgr().MapSymbolicDimEqual(symDimS0, symDimS1);
  shapeAnalysis.symbolicDimMgr().MapSymbolicDimEqual(symDimS0, symDimS2);

  EXPECT_TRUE(shapeAnalysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shapeAnalysis.IsShapeEqual(value1, value5));
}

TEST(shape_op, tensor_dim) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::dialect::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::Operation *op = CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::OpResult resDenseTensorValue = op->result(0);

  pir::dialect::TensorDimOp tensorDimOp0 =
      builder.Build<pir::dialect::TensorDimOp>(resDenseTensorValue, 0);
  pir::OpResult res0 = tensorDimOp0.out();

  pir::OpResult indexValue =
      builder
          .Build<pir::ConstantOp>(
              pir::Int64Attribute::get(pir::IrContext::Instance(), 1),
              pir::IndexType::get(pir::IrContext::Instance()))
          ->result(0);
  pir::dialect::TensorDimOp tensorDimOp1 =
      builder.Build<pir::dialect::TensorDimOp>(resDenseTensorValue, indexValue);
  pir::OpResult res1 = tensorDimOp1.out();

  EXPECT_EQ(res0.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(res1.type(), pir::IndexType::get(ctx));
  EXPECT_EQ(tensorDimOp0.source(), resDenseTensorValue);
  EXPECT_EQ(tensorDimOp1.source(), resDenseTensorValue);
  EXPECT_EQ(tensorDimOp1.index(), indexValue);
}
