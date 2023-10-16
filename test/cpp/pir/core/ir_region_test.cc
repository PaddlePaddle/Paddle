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

#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"

TEST(region, erase_op_test) {
  // (1) Init environment.
  pir::IrContext* ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  // (3) Def a = ConstantOp("2.0"); b = ConstantOp("2.0");
  pir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  pir::Float32Type fp32_type = builder.float32_type();
  pir::OpResult a =
      builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);
  pir::OpResult b =
      builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);

  // (6) Def c = CombineOp(a, b)
  builder.Build<pir::CombineOp>(std::vector<pir::Value>{a, b});

  // Test pir::Block::erase
  pir::Block* block = program.block();
  EXPECT_EQ(block->size(), 3u);
  block->erase(*(block->back()));
  EXPECT_EQ(block->size(), 2u);

  // Test pir::Region::erase
  pir::Region& region = program.module_op()->region(0);
  region.push_back(new pir::Block());
  EXPECT_EQ(region.size(), 2u);
  region.erase(region.begin());
  EXPECT_EQ(region.size(), 1u);
}
