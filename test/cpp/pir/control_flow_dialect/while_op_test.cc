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

#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

using namespace paddle::dialect;  // NOLINT

// example for while_op use
// while(i < ten) { i = i + 1;}
TEST(while_op_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto i =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 1, phi::DataType::INT32)
          .out();
  auto ten =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 10, phi::DataType::INT32)
          .out();

  // comput condition value: i < ten
  auto cond_value = builder.Build<LessThanOp>(i, ten).out();

  auto while_op =
      builder.Build<WhileOp>(cond_value, std::vector<pir::Value>{i, ten});

  // { i = i + 1}
  pir::Block* body_block = while_op.body_block();
  auto body_i_argument = body_block->AddArgument(i.type());
  auto body_ten_argument = body_block->AddArgument(ten.type());
  builder.SetInsertionPointToStart(body_block);
  auto one =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 1, phi::DataType::INT32)
          .out();
  auto new_i = builder.Build<AddOp>(body_i_argument, one).out();

  // comput new condition value: new_i < new_ten
  auto new_cond_value =
      builder.Build<LessThanOp>(new_i, body_ten_argument).out();

  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{new_cond_value, new_i, body_ten_argument});

  builder.SetInsertionPointAfter(while_op);
  std::stringstream ss;
  program.Print(ss);

  LOG(INFO) << ss.str();

  EXPECT_EQ(while_op.cond(), cond_value);
}
