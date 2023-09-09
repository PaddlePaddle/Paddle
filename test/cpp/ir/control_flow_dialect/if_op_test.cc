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
#include <iostream>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/ir/dialect/control_flow/ir/cf_ops.h"

TEST(if_op_test, base) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::ControlFlowDialect>();

  ir::Program program(ctx);
  ir::Block* block = program.block();
  ir::Builder builder(ctx, block);

  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::vector<ir::Type>{builder.bool_type()});

  ir::Block* true_block = if_op.true_block();

  builder.SetInsertionPointToStart(true_block);

  auto full_op_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2}, true, phi::DataType::BOOL);
  builder.Build<ir::YieldOp>(std::vector<ir::OpResult>{full_op_1.out()});

  ir::Block* false_block = if_op.false_block();

  builder.SetInsertionPointToStart(false_block);

  auto full_op_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{3}, true, phi::DataType::BOOL);
  builder.Build<ir::YieldOp>(std::vector<ir::OpResult>{full_op_2.out()});

  std::stringstream ss;
  program.Print(ss);

  LOG(INFO) << ss.str();
}
