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
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/verify.h"

TEST(ir_op_info_test, op_op_info_test) {
  pir::IrContext* context = pir::IrContext::Instance();
  pir::Program program(context);

  pir::Block* block = program.block();
  pir::Builder builder(context, block);
  builder.Build<pir::ConstantOp>(pir::Int32Attribute::get(context, 5),
                                 pir::Int32Type::get(context));

  auto& op = block->back();

  EXPECT_EQ(block->end(), ++pir::Block::Iterator(op));

  auto& info_map = context->registered_op_info_map();
  EXPECT_FALSE(info_map.empty());

  void* info_1 = op.info();
  auto info_2 = pir::OpInfo::RecoverFromVoidPointer(info_1);
  EXPECT_EQ(op.info(), info_2);
  pir::Verify(program.module_op());
}
