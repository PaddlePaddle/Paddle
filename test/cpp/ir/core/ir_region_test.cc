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

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"

TEST(region, erase_op_test) {
  // (1) Init environment.
  ir::IrContext* ctx = ir::IrContext::Instance();

  // (2) Create an empty program object
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder(ctx, program.block());

  // (3) Def a = ConstantOp("2.0"); b = ConstantOp("2.0");
  ir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  ir::Float32Type fp32_type = builder.float32_type();
  ir::OpResult a = builder.Build<ir::ConstantOp>(fp_attr, fp32_type)->result(0);
  ir::OpResult b = builder.Build<ir::ConstantOp>(fp_attr, fp32_type)->result(0);

  // (6) Def c = CombineOp(a, b)
  builder.Build<ir::CombineOp>(std::vector<ir::OpResult>{a, b});

  // Test ir::Block::erase
  ir::Block* block = program.block();
  EXPECT_EQ(block->size(), 3u);
  block->erase(*(block->back()));
  EXPECT_EQ(block->size(), 2u);

  // Test ir::Region::erase
  ir::Region& region = program.module_op()->region(0);
  region.push_back(new ir::Block());
  EXPECT_EQ(region.size(), 2u);
  region.erase(region.begin());
  EXPECT_EQ(region.size(), 1u);
}
