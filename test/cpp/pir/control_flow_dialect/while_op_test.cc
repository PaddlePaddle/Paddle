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

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(less_than, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_n, CPU, ALL_LAYOUT);

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

  // compute condition value: i < ten
  auto cond_value = builder.Build<LessThanOp>(i, ten).out();

  auto while_op =
      builder.Build<WhileOp>(cond_value, std::vector<pir::Value>{i, ten});

  // { i = i + 1}
  pir::Block& body_block = while_op.body();
  auto body_i_argument = body_block.arg(0);
  auto body_ten_argument = body_block.arg(1);
  builder.SetInsertionPointToStart(&body_block);
  auto one =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 1, phi::DataType::INT32)
          .out();
  auto new_i = builder.Build<AddOp>(body_i_argument, one).out();

  // compute new condition value: new_i < new_ten
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
  ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);
  auto i =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 0, phi::DataType::INT32)
          .out();
  auto ten =
      builder.Build<FullOp>(std::vector<int64_t>{10}, 10, phi::DataType::INT32)
          .out();
  auto x = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  auto y = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 2.0f).out();
  auto one =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 1, phi::DataType::INT32)
          .out();
  // def cond(i, x):
  //   return i < 10
  // def body(i, x):
  //   return i + 1, x + y
  // }
  auto cond_value = builder.Build<LessThanOp>(i, ten).out();

  auto [stack, inlet, outlet] = builder.Build<pir::StackCreateOp>().out();
  (void)(stack);
  auto while_op =
      builder.Build<WhileOp>(cond_value, std::vector<pir::Value>{i, x});

  // { return i + 1, x + y}
  auto& body_block = while_op.body();
  builder.SetInsertionPointToStart(&body_block);

  auto body_i_argument = body_block.arg(0);
  auto body_x_argument = body_block.arg(1);

  auto new_i = builder.Build<AddOp>(body_i_argument, one).out();
  auto new_x = builder.Build<AddOp>(body_x_argument, y).out();

  // compute new condition value: new_i < new_ten
  auto new_cond_value = builder.Build<LessThanOp>(new_i, ten).out();

  builder.Build<pir::TuplePushOp>(
      inlet, std::initializer_list<pir::Value>{body_x_argument});

  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{new_cond_value, new_i, new_x});

  builder.SetInsertionPointAfter(while_op);

  auto i_out = while_op->result(0);
  auto x_out = while_op->result(1);
  EXPECT_EQ(i_out.type(), i.type());
  EXPECT_EQ(x_out.type(), x.type());

  // build backward network
  auto x_out_grad =
      builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  auto zero = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 0.0).out();

  // the input of  while_grad op is {x_out_grad, zero}
  // the output of while_grad op is {x_grad, y_grad}
  // the value {i , one, ten} is stop gradient.

  auto bwd_cond = builder.Build<HasElementsOp>(stack).out();
  auto while_grad = builder.Build<WhileOp>(
      bwd_cond, std::vector<pir::Value>{x_out_grad, zero});
  pir::Block& bwd_body_block = while_grad.body();
  builder.SetInsertionPointToStart(&bwd_body_block);
  auto local_x_out_grad_arg = bwd_body_block.arg(0);
  auto local_y_grad_arg = bwd_body_block.arg(1);

  auto pop_op = builder.Build<pir::TuplePopOp>(outlet);
  auto bwd_body_x_argument = pop_op.outlet_element(0);

  auto add_grad_op =
      builder.Build<AddGradOp>(bwd_body_x_argument, y, local_x_out_grad_arg);
  auto bwd_body_x_argument_grad = add_grad_op.x_grad();
  auto local_y_grad = add_grad_op.y_grad();

  // accumulate gradient
  auto combine_y = builder
                       .Build<pir::CombineOp>(std::vector<pir::Value>{
                           local_y_grad, local_y_grad_arg})
                       .out();
  auto local_next_y_grad = builder.Build<AddNOp>(combine_y).out();

  auto next_bwd_cond = builder.Build<HasElementsOp>(stack).out();
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{
      next_bwd_cond, bwd_body_x_argument_grad, local_next_y_grad});

  auto x_grad = while_grad.result(0);
  auto y_grad = while_grad.result(1);

  EXPECT_EQ(x_grad.type(), x.type());
  EXPECT_EQ(y_grad.type(), y.type());

  LOG(INFO) << program;

  auto place = phi::CPUPlace();
#ifdef PADDLE_WITH_CUDA
  place = phi::GPUPlace(0);
#endif
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program, place);
  paddle::framework::Scope scope;
  paddle::framework::InterpreterCore test_core(
      place, {}, kernel_program->block(), &scope);
  test_core.Run({});
}
