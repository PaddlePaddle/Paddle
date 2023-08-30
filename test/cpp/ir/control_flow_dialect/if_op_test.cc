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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"

TEST(if_op_test, base) {
  ir::IrContext ctx;
  ctx.GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  ir::Program program(&ctx);
  ir::Block* block = program.block();
  ir::Builder builder(&ctx, block);

  std::vector<int64_t> shape{1};
  auto full_op =
      builder.Build<paddle::dialect::FullOp>(shape, true, phi::DataType::BOOL);
  builder.Build<paddle::dialect::IfOp>(full_op.out(), std::vector<ir::Type>{});
  program.Print(std::cout);
}
