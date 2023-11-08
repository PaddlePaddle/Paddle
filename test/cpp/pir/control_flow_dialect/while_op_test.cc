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
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

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
  LOG(INFO) << program;

  EXPECT_EQ(while_op.cond(), cond_value);
}

TEST(while_op_test, network_with_backward) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);
  auto x = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  auto y = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 2.0f).out();
  auto cond = builder.Build<LessThanOp>(x, y).out();
  auto [stack_0, inlet_0, outlet_0] = builder.Build<pir::CreateStackOp>().out();
  auto [stack_1, inlet_1, outlet_1] = builder.Build<pir::CreateStackOp>().out();
  (void)(stack_0);
  (void)(stack_1);

  auto if_op = builder.Build<IfOp>(cond, std::vector<pir::Type>{x.type()});

  builder.SetInsertionPointToStart(if_op.true_block());
  auto local1_z = builder.Build<AddOp>(x, y).out();
  auto local1_w = builder.Build<AddOp>(local1_z, y).out();
  builder.Build<pir::PushBackOp>(inlet_0,
                                 std::initializer_list<pir::Value>{local1_z});
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{local1_w});

  builder.SetInsertionPointToStart(if_op.false_block());
  auto local2_z = builder.Build<MatmulOp>(x, y).out();
  auto local2_w = builder.Build<MatmulOp>(local2_z, y).out();
  builder.Build<pir::PushBackOp>(inlet_1,
                                 std::initializer_list<pir::Value>{local2_z});
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{local2_w});

  builder.SetInsertionPointToEnd(block);

  // build backward network
  auto out_grad = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  // the output of if_grad op is {x_grad, y_grad}
  auto if_grad =
      builder.Build<IfOp>(cond, std::vector<pir::Type>{x.type(), y.type()});

  // construct the true block of if_grad
  builder.SetInsertionPointToStart(if_grad.true_block());
  auto pop_local1_z = builder.Build<pir::PopBackOp>(outlet_0).outlet_element(0);
  auto local1_add_grad_op = builder.Build<AddGradOp>(pop_local1_z, y, out_grad);
  auto pop_local1_z_grad = local1_add_grad_op.x_grad(),
       local1_y_grad_0 = local1_add_grad_op.y_grad();
  auto local1_add_grad_op_1 = builder.Build<AddGradOp>(x, y, pop_local1_z_grad);
  auto local1_x_grad = local1_add_grad_op_1.x_grad(),
       local1_y_grad_1 = local1_add_grad_op_1.y_grad();
  auto local1_y_grad =
      builder.Build<AddOp>(local1_y_grad_0, local1_y_grad_1).out();
  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{local1_x_grad, local1_y_grad});

  // construct the false block of if_grad
  builder.SetInsertionPointToStart(if_grad.false_block());
  auto pop_local2_z = builder.Build<pir::PopBackOp>(outlet_1).outlet_element(0);
  auto local2_matmul_grad_op =
      builder.Build<MatmulGradOp>(pop_local2_z, y, out_grad);
  auto pop_local2_z_grad = local2_matmul_grad_op.x_grad(),
       local2_y_grad_0 = local2_matmul_grad_op.y_grad();
  auto local2_matmul_grad_op_1 =
      builder.Build<MatmulGradOp>(x, y, pop_local2_z_grad);
  auto local2_x_grad = local2_matmul_grad_op_1.x_grad(),
       local2_y_grad_1 = local2_matmul_grad_op_1.y_grad();

  auto local2_y_grad =
      builder.Build<AddOp>(local2_y_grad_0, local2_y_grad_1).out();
  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{local2_x_grad, local2_y_grad});

  builder.SetInsertionPointToEnd(block);

  LOG(INFO) << program;
}
