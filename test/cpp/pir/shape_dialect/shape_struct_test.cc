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
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(shape_struct_test, symbolic_dim_product) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::shape::SymbolicDimOp sym_dim = builder.Build<pir::shape::SymbolicDimOp>(
      "S0", pir::ShapedTypeInterface::kDynamic, false, false, false, false);
  pir::SymbolicDimProduct sym_dim_product1;
  pir::SymbolicDimProduct sym_dim_product2;
  sym_dim_product1.symbols.push_back(sym_dim);
  sym_dim_product1.factor *= 10;
  EXPECT_EQ(sym_dim_product1.factor, 10);
  EXPECT_NE(sym_dim_product1, sym_dim_product2);
  EXPECT_FALSE(sym_dim_product1.empty());
}

TEST(shape_struct_test, symbolic_dim_table) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::shape::SymbolicDimOp sym_dim = builder.Build<pir::shape::SymbolicDimOp>(
      "S0", 10, false, false, false, false);

  pir::SymbolTable symbol_table(program.module_op());
  EXPECT_EQ(symbol_table.insert(sym_dim), "S0");
  EXPECT_EQ(symbol_table.Lookup<pir::shape::SymbolicDimOp>("S0"), sym_dim);
  EXPECT_EQ(symbol_table.getOp(), program.module_op());
  EXPECT_FALSE(symbol_table.Lookup<pir::shape::SymbolicDimOp>("S1"));
}

TEST(shape_struct_test, symbolic_dim_mgr_simple) {
  /******************************************************/
  /* Mgr simple version, only SymbolicDimOp related func. */
  /******************************************************/
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::SymbolicDimMgr sym_dim_mgr(program.module_op());
  pir::shape::SymbolicDimOp sym_dim_s0 = sym_dim_mgr.NewSymbolicDim();
  pir::shape::SymbolicDimOp sym_dim_s1 = sym_dim_mgr.NewSymbolicDim();
  pir::shape::SymbolicDimOp sym_dim_c10 =
      sym_dim_mgr.NewConstantSymbolicDim(10);
  sym_dim_mgr.MapSymbolicDimEqual(sym_dim_s0, sym_dim_s1);

  auto op = test::CreateDenseTensorOp(
      ctx, {pir::ShapedTypeInterface::kDynamic, 2}, {"op_attr"}, {"op_name"});
  pir::Value res = op->result(0);

  std::vector<pir::shape::SymbolicDimOp> sym_dim_vec =
      sym_dim_mgr.CreateSymbolicDimsForRankedValue(res);

  EXPECT_EQ(sym_dim_s0.GetSymName(), "S0");
  EXPECT_EQ(sym_dim_s1.GetSymName(), "S1");
  EXPECT_EQ(sym_dim_s1.GetDimSize(), pir::ShapedTypeInterface::kDynamic);
  EXPECT_EQ(sym_dim_c10.GetSymName(), "C10");
  EXPECT_EQ(sym_dim_c10.GetDimSize(), 10);
  EXPECT_EQ(sym_dim_vec[0].GetSymName(), "S2");
  EXPECT_EQ(sym_dim_vec[1].GetSymName(), "C2");
  EXPECT_EQ(sym_dim_mgr.symbolTable().Lookup<pir::shape::SymbolicDimOp>("S0"),
            sym_dim_s0);
  EXPECT_EQ(sym_dim_mgr.symbolTable().Lookup<pir::shape::SymbolicDimOp>("C10"),
            sym_dim_c10);
  EXPECT_EQ(sym_dim_mgr.GetRootSymbolicDim(sym_dim_s1), sym_dim_s0);
  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimEqual(sym_dim_s0, sym_dim_s1));
  EXPECT_FALSE(sym_dim_mgr.IsSymbolicDimEqual(sym_dim_s0, sym_dim_c10));
}

TEST(shape_struct_test, symbolic_dim_mgr_complex) {
  /***************************************************************/
  /* Mgr with constraintOp, and SymbolicDimProduct related func. */
  /***************************************************************/
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::SymbolicDimMgr sym_dim_mgr(program.module_op());
  auto func_op =
      sym_dim_mgr.symbolTable().getOp()->dyn_cast<pir::shape::FuncOp>();

  pir::Builder builder = pir::Builder(ctx, func_op.block());

  pir::shape::SymbolicDimOp sym_dim_s0 = sym_dim_mgr.NewSymbolicDim("S0");
  pir::shape::SymbolicDimOp sym_dim_s1 = sym_dim_mgr.NewSymbolicDim("S1");
  pir::shape::SymbolicDimOp sym_dim_s2 = sym_dim_mgr.NewSymbolicDim("S2");
  pir::shape::SymbolicDimOp sym_dim_s3 = sym_dim_mgr.NewSymbolicDim("S3");
  pir::shape::SymbolicDimOp sym_dim_s4 = sym_dim_mgr.NewSymbolicDim("S4");
  pir::shape::SymbolicDimOp sym_dim_s5 = sym_dim_mgr.NewSymbolicDim("S5");
  pir::shape::SymbolicDimOp sym_dim_s6 = sym_dim_mgr.NewSymbolicDim("S6");
  pir::shape::SymbolicDimOp sym_dim_s7 = sym_dim_mgr.NewSymbolicDim("S7");
  pir::shape::SymbolicDimOp sym_dim_s8 = sym_dim_mgr.NewSymbolicDim("S8");
  pir::shape::SymbolicDimOp sym_dim_s9 = sym_dim_mgr.NewSymbolicDim("S9");
  pir::shape::SymbolicDimOp sym_dim_s10 = sym_dim_mgr.NewSymbolicDim("S10");
  pir::shape::SymbolicDimOp sym_dim_s11 = sym_dim_mgr.NewSymbolicDim("S11");
  pir::shape::SymbolicDimOp sym_dim_s12 = sym_dim_mgr.NewSymbolicDim("S12");
  pir::shape::SymbolicDimOp sym_dim_c10 =
      sym_dim_mgr.NewConstantSymbolicDim(10);
  pir::shape::SymbolicDimOp sym_dim_c20 =
      sym_dim_mgr.NewConstantSymbolicDim(20);

  pir::OpResult dim_op_s0 = builder.Build<pir::shape::DimOp>("S0").out();
  pir::OpResult dim_op_s1 = builder.Build<pir::shape::DimOp>("S1").out();
  pir::OpResult dim_op_s2 = builder.Build<pir::shape::DimOp>("S2").out();
  pir::OpResult dim_op_s3 = builder.Build<pir::shape::DimOp>("S3").out();
  pir::OpResult dim_op_s4 = builder.Build<pir::shape::DimOp>("S4").out();
  pir::OpResult dim_op_s5 = builder.Build<pir::shape::DimOp>("S5").out();
  pir::OpResult dim_op_s6 = builder.Build<pir::shape::DimOp>("S6").out();
  pir::OpResult dim_op_s7 = builder.Build<pir::shape::DimOp>("S7").out();
  pir::OpResult dim_op_s8 = builder.Build<pir::shape::DimOp>("S8").out();
  pir::OpResult dim_op_s9 = builder.Build<pir::shape::DimOp>("S9").out();
  pir::OpResult dim_op_s10 = builder.Build<pir::shape::DimOp>("S10").out();
  pir::OpResult dim_op_s11 = builder.Build<pir::shape::DimOp>("S11").out();
  pir::OpResult dim_op_c10 = builder.Build<pir::shape::DimOp>("C10").out();
  pir::OpResult dim_op_c20 = builder.Build<pir::shape::DimOp>("C20").out();
  pir::OpResult constant =
      builder
          .Build<pir::ConstantOp>(pir::Int32Attribute::get(ctx, 2),
                                  pir::Int32Type::get(ctx))
          ->result(0);

  // Mark S1 == S2.
  builder.Build<pir::shape::TieProductEqualOp>(
      2, 2, std::vector<pir::Value>{constant, dim_op_s1, dim_op_s2, constant});
  // Mark S0 * S1 == S2 * S3, For check S0 == S3.
  builder.Build<pir::shape::TieProductEqualOp>(
      2,
      2,
      std::vector<pir::Value>{dim_op_s0, dim_op_s1, dim_op_s2, dim_op_s3});
  // Mark S4 * S0 * S1 == S2 * S3 * S5, For check S4 == S5.
  builder.Build<pir::shape::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::Value>{
          dim_op_s4, dim_op_s0, dim_op_s1, dim_op_s2, dim_op_s3, dim_op_s5});
  // For check S6 == C10 * C20.
  builder.Build<pir::shape::TieProductEqualOp>(
      1, 2, std::vector<pir::Value>{dim_op_s6, dim_op_c10, dim_op_c20});
  // Mark C10 * S0 * S1 == S2 * S3 * S7, for check C10 == S7.
  builder.Build<pir::shape::TieProductEqualOp>(
      3,
      3,
      std::vector<pir::Value>{
          dim_op_c10, dim_op_s0, dim_op_s1, dim_op_s2, dim_op_s3, dim_op_s7});

  // For unsimplify product case: S8 * S9 == S10 * S11
  builder.Build<pir::shape::TieProductEqualOp>(
      2,
      2,
      std::vector<pir::Value>{dim_op_s8, dim_op_s9, dim_op_s10, dim_op_s11});

  auto op = test::CreateDenseTensorOp(
      ctx, {-1, -1, -1, -1, -1, -1}, {"op0_attr"}, {"op0_name"});
  auto op_ = test::CreateDenseTensorOp(
      ctx, {-1, -1, -1, -1, -1, 10, 20}, {"op1_attr"}, {"op1_name"});
  pir::OpResult res = op->result(0);
  pir::OpResult res_ = op_->result(0);

  builder.SetInsertionPointToBlockEnd(program.block());
  pir::shape::TieShapeOp tie_shape_op1 =
      builder.Build<pir::shape::TieShapeOp>(res);
  pir::shape::TieShapeOp tie_shape_op2 =
      builder.Build<pir::shape::TieShapeOp>(res_);

  pir::Attribute attr_s0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attr_s1 = pir::StrAttribute::get(ctx, "S1");
  pir::Attribute attr_s2 = pir::StrAttribute::get(ctx, "S2");
  pir::Attribute attr_s3 = pir::StrAttribute::get(ctx, "S3");
  pir::Attribute attr_s4 = pir::StrAttribute::get(ctx, "S4");
  pir::Attribute attr_s5 = pir::StrAttribute::get(ctx, "S5");
  pir::Attribute attr_s6 = pir::StrAttribute::get(ctx, "S6");
  pir::Attribute attr_s7 = pir::StrAttribute::get(ctx, "S7");
  pir::Attribute attr_s8 = pir::StrAttribute::get(ctx, "S8");
  pir::Attribute attr_s9 = pir::StrAttribute::get(ctx, "S9");
  pir::Attribute attr_s10 = pir::StrAttribute::get(ctx, "S10");
  pir::Attribute attr_s11 = pir::StrAttribute::get(ctx, "S11");
  pir::Attribute attr_c10 = pir::StrAttribute::get(ctx, "C10");
  pir::Attribute attr_c20 = pir::StrAttribute::get(ctx, "C20");

  std::vector<pir::Attribute> new_attrs1 = {
      attr_s0, attr_s1, attr_s2, attr_s3, attr_s4, attr_s5};
  std::vector<pir::Attribute> new_attrs2 = {attr_s6,
                                            attr_s7,
                                            attr_s8,
                                            attr_s9,
                                            attr_s10,
                                            attr_s11,
                                            attr_c10,
                                            attr_c20};
  std::vector<pir::Attribute> new_attrs_ref = {
      attr_s0, attr_s1, attr_s1, attr_s0, attr_s2, attr_s2};

  auto array_attr1 = pir::ArrayAttribute::get(ctx, new_attrs1);
  auto array_attr2 = pir::ArrayAttribute::get(ctx, new_attrs2);
  auto array_attr_ref = pir::ArrayAttribute::get(ctx, new_attrs_ref);

  tie_shape_op1->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), array_attr1);
  tie_shape_op2->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), array_attr2);

  EXPECT_TRUE(sym_dim_mgr.Load());

  // For check indirect equality: S1 * S4 == S2 * S5
  pir::SymbolicDimProduct sym_dim_product_lhs1;
  pir::SymbolicDimProduct sym_dim_product_rhs1;

  sym_dim_product_lhs1.symbols.push_back(sym_dim_s1);
  sym_dim_product_lhs1.symbols.push_back(sym_dim_s4);

  sym_dim_product_rhs1.symbols.push_back(sym_dim_s2);
  sym_dim_product_rhs1.symbols.push_back(sym_dim_s5);

  // For uncompletely simplied product check: S8 * S9 * S12 == S10 * S11 * S12
  pir::SymbolicDimProduct sym_dim_product_lhs2;
  pir::SymbolicDimProduct sym_dim_product_rhs2;

  sym_dim_product_lhs2.symbols.push_back(sym_dim_s8);
  sym_dim_product_lhs2.symbols.push_back(sym_dim_s9);
  sym_dim_product_lhs2.symbols.push_back(sym_dim_s12);

  sym_dim_product_rhs2.symbols.push_back(sym_dim_s10);
  sym_dim_product_rhs2.symbols.push_back(sym_dim_s11);
  sym_dim_product_rhs2.symbols.push_back(sym_dim_s12);

  // For check SimplifySymbolicDimProduct, {factor = 1, Sym = {S7}} => {factor =
  // 10}
  pir::SymbolicDimProduct sym_dim_product_s7;
  sym_dim_product_s7.symbols.push_back(sym_dim_s7);
  pir::SymbolicDimProduct simplified_product_s7 =
      sym_dim_mgr.SimplifySymbolicDimProduct(sym_dim_product_s7);

  // For check SimplifySymbolicDimProductPair, X * Y * Y, Y * Y * Z => X, Z
  pir::SymbolicDimProduct sym_dim_product_pair_lhs;
  pir::SymbolicDimProduct sym_dim_product_pair_rhs;
  pir::SymbolicDimProduct new_lhs, new_rhs;
  sym_dim_product_pair_lhs.symbols.push_back(sym_dim_s4);
  sym_dim_product_pair_lhs.symbols.push_back(sym_dim_s1);
  sym_dim_product_pair_lhs.symbols.push_back(sym_dim_s2);
  sym_dim_product_pair_rhs.symbols.push_back(sym_dim_s1);
  sym_dim_product_pair_rhs.symbols.push_back(sym_dim_s2);
  sym_dim_product_pair_rhs.symbols.push_back(sym_dim_s3);

  std::tie(new_lhs, new_rhs) = sym_dim_mgr.SimplifySymbolicDimProductPair(
      sym_dim_product_pair_lhs, sym_dim_product_pair_rhs);

  // For check SymbolicDimProductDivide, {S4 * S1 * C20} / {S1 * C10} => {factor
  // = 2 Sym = {S4}}
  pir::SymbolicDimProduct sym_dim_product_div_lhs;
  pir::SymbolicDimProduct sym_dim_product_div_rhs;
  sym_dim_product_div_lhs.symbols.push_back(sym_dim_s4);
  sym_dim_product_div_lhs.symbols.push_back(sym_dim_s1);
  sym_dim_product_div_lhs.symbols.push_back(sym_dim_c20);
  sym_dim_product_div_rhs.symbols.push_back(sym_dim_s1);
  sym_dim_product_div_rhs.symbols.push_back(sym_dim_c10);

  pir::SymbolicDimProduct *divRes = sym_dim_mgr.SymbolicDimProductDivide(
      sym_dim_product_div_lhs, sym_dim_product_div_rhs);

  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimEqual(sym_dim_s1, sym_dim_s2));
  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimEqual(sym_dim_s0, sym_dim_s3));
  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimEqual(sym_dim_s4, sym_dim_s5));
  EXPECT_EQ(sym_dim_s6.GetDimSize(), 200);
  EXPECT_EQ(sym_dim_mgr.symbolTable().Lookup<pir::shape::SymbolicDimOp>("C20"),
            sym_dim_c20);
  EXPECT_EQ(sym_dim_s7.GetDimSize(), sym_dim_c10.GetDimSize());
  EXPECT_EQ(simplified_product_s7.factor, 10);
  EXPECT_EQ(simplified_product_s7.symbols.size(), static_cast<size_t>(0));
  EXPECT_EQ(new_lhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(new_rhs.symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(new_lhs.symbols[0], sym_dim_mgr.GetRootSymbolicDim(sym_dim_s4));
  EXPECT_EQ(new_rhs.symbols[0], sym_dim_mgr.GetRootSymbolicDim(sym_dim_s3));
  EXPECT_EQ(divRes->factor, 2);
  EXPECT_EQ(divRes->symbols.size(), static_cast<size_t>(1));
  EXPECT_EQ(divRes->symbols[0], sym_dim_mgr.GetRootSymbolicDim(sym_dim_s4));
  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimProductEqual(sym_dim_product_lhs1,
                                                    sym_dim_product_rhs1));
  EXPECT_TRUE(sym_dim_mgr.IsSymbolicDimProductEqual(sym_dim_product_lhs2,
                                                    sym_dim_product_rhs2));
  EXPECT_TRUE(sym_dim_mgr.Save());

  pir::SymbolicDimMgr sym_dim_mgr_new(program.module_op());
  EXPECT_TRUE(sym_dim_mgr_new.Load());

  auto attrs = tie_shape_op1.attribute<pir::ArrayAttribute>(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName());
  EXPECT_FALSE(
      sym_dim_mgr_new.symbolTable().Lookup<pir::shape::SymbolicDimOp>("S7"));
  EXPECT_EQ(sym_dim_mgr_new.symbolTable()
                .Lookup<pir::shape::TieProductEqualOp>("tie_product_equal")
                .size(),
            static_cast<size_t>(1));

  EXPECT_EQ(attrs.AsVector(), array_attr_ref.AsVector());
}

TEST(shape_struct_test, shape_analysis) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::shape::FuncOp func_op = builder.Build<pir::shape::FuncOp>();

  phi::DDim dims_D_2 = {-1, 2};
  phi::DDim dims_2_2 = {2, 2};
  phi::DDim dims_D = {-1};

  // same shape with dynamic: value1 == value2
  auto op1 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op1_attr"}, {"op1_name"});
  auto op2 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op2_attr"}, {"op2_name"});
  pir::OpResult value1 = op1->result(0);
  pir::OpResult value2 = op2->result(0);

  // same shape with static: value3 == value4
  auto op3 =
      test::CreateDenseTensorOp(ctx, dims_2_2, {"op3_attr"}, {"op3_name"});
  auto op4 =
      test::CreateDenseTensorOp(ctx, dims_2_2, {"op4_attr"}, {"op4_name"});
  pir::OpResult value3 = op3->result(0);
  pir::OpResult value4 = op4->result(0);

  // one dimension with dynamic: value5 != value1 != value3
  auto op5 = test::CreateDenseTensorOp(ctx, dims_D, {"op5_attr"}, {"op5_name"});
  pir::OpResult value5 = op5->result(0);

  pir::shape::TieShapeOp tie_shape_op1 =
      builder.Build<pir::shape::TieShapeOp>(value1);
  pir::shape::TieShapeOp tie_shape_op2 =
      builder.Build<pir::shape::TieShapeOp>(value2);
  pir::shape::TieShapeOp tie_shape_op3 =
      builder.Build<pir::shape::TieShapeOp>(value3);
  pir::shape::TieShapeOp tie_shape_op4 =
      builder.Build<pir::shape::TieShapeOp>(value4);
  pir::shape::TieShapeOp tie_shape_op5 =
      builder.Build<pir::shape::TieShapeOp>(value5);

  builder.SetInsertionPointToBlockEnd(func_op.block());
  builder.Build<pir::shape::SymbolicDimOp>("C2", 2, true, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s0 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S0", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s1 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S1", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s2 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S2", pir::ShapedTypeInterface::kDynamic, false, false, true, true);

  pir::Attribute attr_s0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attr_s1 = pir::StrAttribute::get(ctx, "S1");
  pir::Attribute attr_s2 = pir::StrAttribute::get(ctx, "S2");
  pir::Attribute attr_c2 = pir::StrAttribute::get(ctx, "C2");

  auto attr_op1 = pir::ArrayAttribute::get(ctx, {attr_s0, attr_c2});
  auto attr_op2 = pir::ArrayAttribute::get(ctx, {attr_s1, attr_c2});
  auto attr_op3 = pir::ArrayAttribute::get(ctx, {attr_c2, attr_c2});
  auto attr_op4 = pir::ArrayAttribute::get(ctx, {attr_c2, attr_c2});
  auto attr_op5 = pir::ArrayAttribute::get(ctx, {attr_s2});

  tie_shape_op1->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op1);
  tie_shape_op2->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op2);
  tie_shape_op3->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op3);
  tie_shape_op4->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op4);
  tie_shape_op5->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op5);

  pir::ShapeConstraintIRAnalysis shape_analysis(program.module_op());
  EXPECT_TRUE(shape_analysis.IsShapeEqual(value3, value4));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value3));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value5));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value3, value5));
  EXPECT_TRUE(shape_analysis.IsProductEqual(value1, {1}, value3, {0}));
  EXPECT_TRUE(shape_analysis.IsSameNumElements(value4, value3));

  shape_analysis.symbolicDimMgr().MapSymbolicDimEqual(sym_dim_s0, sym_dim_s1);
  shape_analysis.symbolicDimMgr().MapSymbolicDimEqual(sym_dim_s0, sym_dim_s2);

  const auto &val_sym_dim1 =
      shape_analysis.GetOrCreateSymbolicDimsForRankedValue(value1);
  const auto &val_sym_dim2 =
      shape_analysis.GetOrCreateSymbolicDimsForRankedValue(value2);
  EXPECT_TRUE(shape_analysis.symbolicDimMgr().IsSymbolicDimEqual(
      val_sym_dim1[0], val_sym_dim2[0]));

  EXPECT_TRUE(shape_analysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value5));
}

TEST(shape_struct_test, shape_analysis_manager) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  pir::shape::FuncOp func_op = builder.Build<pir::shape::FuncOp>();

  phi::DDim dims_D_2 = {-1, 2};
  phi::DDim dims_2_2 = {2, 2};
  phi::DDim dims_D = {-1};

  // same shape with dynamic: value1 == value2
  auto op1 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op1_attr"}, {"op1_name"});
  auto op2 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op2_attr"}, {"op2_name"});
  pir::OpResult value1 = op1->result(0);
  pir::OpResult value2 = op2->result(0);

  // same shape with static: value3 == value4
  auto op3 =
      test::CreateDenseTensorOp(ctx, dims_2_2, {"op3_attr"}, {"op3_name"});
  auto op4 =
      test::CreateDenseTensorOp(ctx, dims_2_2, {"op4_attr"}, {"op4_name"});
  pir::OpResult value3 = op3->result(0);
  pir::OpResult value4 = op4->result(0);

  // one dimension with dynamic: value5 != value1 != value3
  auto op5 = test::CreateDenseTensorOp(ctx, dims_D, {"op5_attr"}, {"op5_name"});
  pir::OpResult value5 = op5->result(0);

  pir::shape::TieShapeOp tie_shape_op1 =
      builder.Build<pir::shape::TieShapeOp>(value1);
  pir::shape::TieShapeOp tie_shape_op2 =
      builder.Build<pir::shape::TieShapeOp>(value2);
  pir::shape::TieShapeOp tie_shape_op3 =
      builder.Build<pir::shape::TieShapeOp>(value3);
  pir::shape::TieShapeOp tie_shape_op4 =
      builder.Build<pir::shape::TieShapeOp>(value4);
  pir::shape::TieShapeOp tie_shape_op5 =
      builder.Build<pir::shape::TieShapeOp>(value5);

  builder.SetInsertionPointToBlockEnd(func_op.block());
  builder.Build<pir::shape::SymbolicDimOp>("C2", 2, true, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s0 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S0", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s1 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S1", pir::ShapedTypeInterface::kDynamic, false, false, true, true);
  pir::shape::SymbolicDimOp sym_dim_s2 =
      builder.Build<pir::shape::SymbolicDimOp>(
          "S2", pir::ShapedTypeInterface::kDynamic, false, false, true, true);

  pir::Attribute attr_s0 = pir::StrAttribute::get(ctx, "S0");
  pir::Attribute attr_s1 = pir::StrAttribute::get(ctx, "S1");
  pir::Attribute attr_s2 = pir::StrAttribute::get(ctx, "S2");
  pir::Attribute attr_c2 = pir::StrAttribute::get(ctx, "C2");

  auto attr_op1 = pir::ArrayAttribute::get(ctx, {attr_s0, attr_c2});
  auto attr_op2 = pir::ArrayAttribute::get(ctx, {attr_s1, attr_c2});
  auto attr_op3 = pir::ArrayAttribute::get(ctx, {attr_c2, attr_c2});
  auto attr_op4 = pir::ArrayAttribute::get(ctx, {attr_c2, attr_c2});
  auto attr_op5 = pir::ArrayAttribute::get(ctx, {attr_s2});

  tie_shape_op1->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op1);
  tie_shape_op2->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op2);
  tie_shape_op3->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op3);
  tie_shape_op4->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op4);
  tie_shape_op5->set_attribute(
      pir::shape::SymbolicDimOp::GetSymbolicDimAttrName(), attr_op5);

  auto shape_analysis_mgr = pir::ShapeAnalysisManager::Instance();
  pir::ShapeConstraintIRAnalysis &shape_analysis =
      shape_analysis_mgr.Get(&program);

  EXPECT_TRUE(shape_analysis.IsShapeEqual(value3, value4));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value3));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value5));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value3, value5));
  EXPECT_TRUE(shape_analysis.IsProductEqual(value1, {1}, value3, {0}));
  EXPECT_TRUE(shape_analysis.IsSameNumElements(value4, value3));

  shape_analysis.symbolicDimMgr().MapSymbolicDimEqual(sym_dim_s0, sym_dim_s1);
  shape_analysis.symbolicDimMgr().MapSymbolicDimEqual(sym_dim_s0, sym_dim_s2);

  EXPECT_TRUE(shape_analysis.IsShapeEqual(value1, value2));
  EXPECT_FALSE(shape_analysis.IsShapeEqual(value1, value5));
}
