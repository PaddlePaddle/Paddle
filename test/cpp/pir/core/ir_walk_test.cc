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
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(region, walk_test) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  pir::Float32Type fp32_type = builder.float32_type();
  pir::OpResult a =
      builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);
  pir::OpResult b =
      builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);
  builder.Build<pir::CombineOp>(std::vector<pir::Value>{a, b});
  pir::Type dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  pir::Operation* op =
      test::CreateDenseTensorOp(ctx, dims, {"op_temp"}, {"op_attr"}, dtype);

  // Test pir::Op::Walk
  size_t op_size = 0;
  op->Walk([&](pir::Operation* op) { op_size++; });
  EXPECT_EQ(op_size, 1u);

  // Test pir::Block::Walk
  size_t block_size = 0;
  pir::Block* block = program.block();
  block->Walk([&](pir::Block* block) { block_size++; });
  EXPECT_EQ(block_size, 0u);

  // Test pir::Region::Walk
  size_t region_size = 0;
  pir::Region& region = program.module_op()->region(0);
  region.push_back(new pir::Block());
  region.Walk([&](pir::Region* region) { region_size++; });
  EXPECT_EQ(region_size, 0u);
}
