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

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/utils.h"

TEST(region, erase_op_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  // (3) Def a = ConstantOp("2.0"); b = ConstantOp("2.0");
  pir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  pir::Float32Type fp32_type = builder.float32_type();
  pir::Value a = builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);
  pir::Value b = builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);

  // (6) Def c = CombineOp(a, b)
  builder.Build<pir::CombineOp>(std::vector<pir::Value>{a, b});

  // Test pir::Block::erase
  pir::Block *block = program.block();
  EXPECT_EQ(block->size(), 3u);
  block->erase(block->back());
  EXPECT_EQ(block->size(), 2u);

  // Test pir::Region::erase
  pir::Region &region = program.module_op()->region(0);
  region.push_back(new pir::Block());
  EXPECT_EQ(region.size(), 2u);
  region.erase(region.begin());
  EXPECT_EQ(region.size(), 1u);
}

TEST(region, clone_op_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  // (3) Def a = ConstantOp("2.0"); b = ConstantOp("2.0");
  pir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  pir::Float32Type fp32_type = builder.float32_type();
  pir::Value a = builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);
  pir::Value b = builder.Build<pir::ConstantOp>(fp_attr, fp32_type)->result(0);

  // (6) Def c = CombineOp(a, b)
  builder.Build<pir::CombineOp>(std::vector<pir::Value>{a, b});

  // (7) Test clone module op
  pir::Operation &op = *program.module_op();
  pir::Block &block = op.region(0).front();
  pir::IrMapping mapper;
  pir::Operation &new_op = *op.Clone(mapper, pir::CloneOptions::All());

  // (8) Check the cloned op recursively
  EXPECT_EQ(mapper.Lookup(&op), &new_op);
  EXPECT_EQ(new_op.num_regions(), 1u);
  pir::Region &new_region = new_op.region(0);
  EXPECT_EQ(new_region.size(), 1u);
  pir::Block &new_block = new_region.front();
  EXPECT_EQ(mapper.Lookup(&block), &new_block);
  EXPECT_EQ(new_block.size(), 3u);

  for (auto op_iter = block.begin(), new_op_iter = new_block.begin();
       op_iter != block.end();
       ++op_iter, ++new_op_iter) {
    pir::Operation &op = *op_iter;
    pir::Operation &new_op = *new_op_iter;
    EXPECT_EQ(mapper.Lookup(&op), &new_op);
    EXPECT_EQ(op.num_operands(), new_op.num_operands());
    for (uint32_t i = 0; i < op.num_operands(); ++i) {
      EXPECT_EQ(mapper.Lookup(op.operand_source(i)), new_op.operand_source(i));
    }
    EXPECT_EQ(op.num_results(), new_op.num_results());
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      EXPECT_EQ(mapper.Lookup(op.result(i)), new_op.result(i));
    }
    EXPECT_TRUE(std::equal(op.attributes().begin(),
                           op.attributes().end(),
                           new_op.attributes().begin(),
                           new_op.attributes().end()));
    EXPECT_EQ(op.info(), new_op.info());
  }
}
