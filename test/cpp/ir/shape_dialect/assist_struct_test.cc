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
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/dialect/shape/ir/shape_dialect.h"
#include "paddle/ir/dialect/shape/ir/shape_op.h"

TEST(assist_struct_test, symbolic_dim) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Program program(ctx);
  ctx->GetOrRegisterDialect<ir::dialect::ShapeDialect>();
  ir::Builder builder = ir::Builder(ctx, program.block());
  ir::dialect::SymbolicDim sym_dim = builder.Build<ir::dialect::SymbolicDim>(
      "S0", 10, false, false, false, false);
  EXPECT_EQ(sym_dim.getValue(), 10);
  EXPECT_EQ(sym_dim.getSymName(), "S0");
  EXPECT_FALSE(sym_dim.getKnownNegativeOne());
  EXPECT_FALSE(sym_dim.getKnownNonSizeOne());
  EXPECT_FALSE(sym_dim.getKnownNonSizeZero());
  EXPECT_FALSE(sym_dim.getKnownNonNegative());

  sym_dim.updateValue(20);
  sym_dim.updateSymName("S1");
  sym_dim.updateKnownNegativeOne(true);
  sym_dim.updateKnownNonSizeOne(true);
  sym_dim.updateKnownNonSizeZero(true);
  sym_dim.updateKnownNonNegative(true);

  EXPECT_EQ(sym_dim.getValue(), 20);
  EXPECT_EQ(sym_dim.getSymName(), "S1");
  EXPECT_TRUE(sym_dim.getKnownNegativeOne());
  EXPECT_TRUE(sym_dim.getKnownNonSizeOne());
  EXPECT_TRUE(sym_dim.getKnownNonSizeZero());
  EXPECT_TRUE(sym_dim.getKnownNonNegative());
}
