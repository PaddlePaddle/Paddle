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
