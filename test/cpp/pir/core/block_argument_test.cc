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

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/program.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"

TEST(block_argument_test, base) {
  pir::IrContext ctx;
  ctx.GetOrRegisterDialect<test::TestDialect>();

  pir::Program program(&ctx);
  pir::Block* block = program.block();
  pir::Builder builder(&ctx, block);

  std::vector<pir::Type> types(3, builder.float32_type());
  block->AddArguments(types);

  EXPECT_FALSE(block->args_empty());
  EXPECT_EQ(block->args_size(), types.size());

  uint32_t index = 0;
  for (auto iter = block->args_begin(); iter != block->args_end(); ++iter) {
    EXPECT_EQ(iter->arg_index(), index++);
  }

  pir::Value value = block->argument(0);
  pir::BlockArgument argument = value.dyn_cast<pir::BlockArgument>();
  EXPECT_TRUE(argument);
  EXPECT_EQ(argument.owner(), block);
  EXPECT_EQ(block->argument_type(0), types[0]);
  pir::OpResult op_result = value.dyn_cast<pir::OpResult>();
  EXPECT_FALSE(op_result);

  auto op = builder.Build<pir::ConstantOp>(builder.double_attr(1.0),
                                           builder.float64_type());
  value = op.result(0);
  argument = value.dyn_cast<pir::BlockArgument>();
  EXPECT_FALSE(argument);
  op_result = value.dyn_cast<pir::OpResult>();
  EXPECT_TRUE(op_result);
  block->AddArguments({builder.bool_type()});
  EXPECT_EQ(block->args_size(), 4u);

  value = block->AddArgument(builder.bool_type());
  EXPECT_EQ(value.type(), builder.bool_type());
}
