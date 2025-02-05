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
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(less_than, CPU, ALL_LAYOUT);

using namespace paddle::dialect;  // NOLINT

TEST(if_op_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::vector<pir::Type>{builder.bool_type()});

  auto& true_block = if_op.true_block();

  builder.SetInsertionPointToStart(&true_block);

  auto full_op_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_1.out()});

  auto& false_block = if_op.false_block();

  builder.SetInsertionPointToStart(&false_block);

  auto full_op_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{3}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_2.out()});
  LOG(INFO) << program;
}

TEST(if_op_test, build_by_block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);
  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  // construct true block
  std::unique_ptr<pir::Block> true_block(new pir::Block());
  builder.SetInsertionPointToStart(true_block.get());
  auto full_op_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_1.out()});

  // construct false block
  std::unique_ptr<pir::Block> false_block(new pir::Block());
  builder.SetInsertionPointToStart(false_block.get());
  auto full_op_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{3}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_2.out()});

  builder.SetInsertionPointToBlockEnd(block);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::move(true_block), std::move(false_block));

  EXPECT_FALSE(true_block);
  EXPECT_FALSE(false_block);
  EXPECT_EQ(full_op_2->GetParentProgram(), &program);

  LOG(INFO) << program;

  std::vector<pir::Block*> vec;
  for (auto& block : if_op->blocks()) {
    vec.push_back(&block);
  }
  EXPECT_EQ(vec.size(), 2u);
  EXPECT_EQ(vec[0], &if_op.true_block());
  EXPECT_EQ(vec[1], &if_op.false_block());
  EXPECT_EQ(if_op.num_results(), 1u);
  auto type = if_op.result_type(0).dyn_cast<DenseTensorType>();
  EXPECT_TRUE(type);
  EXPECT_EQ(type.dims(), common::DDim{-1});
}

TEST(if_op_test, network_with_backward) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);
  auto x = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  auto y = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 2.0f).out();
  auto cond = builder.Build<LessThanOp>(x, y).out();
  auto [stack_0, inlet_0, outlet_0] = builder.Build<pir::StackCreateOp>().out();
  auto [stack_1, inlet_1, outlet_1] = builder.Build<pir::StackCreateOp>().out();
  (void)(stack_0);
  (void)(stack_1);

  auto if_op = builder.Build<IfOp>(cond, std::vector<pir::Type>{x.type()});

  builder.SetInsertionPointToStart(&if_op.true_block());

  auto local1_z = builder.Build<AddOp>(x, y).out();
  auto local1_w = builder.Build<AddOp>(local1_z, y).out();
  builder.Build<pir::TuplePushOp>(inlet_0,
                                  std::initializer_list<pir::Value>{local1_z});

  builder.Build<pir::YieldOp>(std::vector<pir::Value>{local1_w});

  builder.SetInsertionPointToStart(&if_op.false_block());
  auto local2_z = builder.Build<MatmulOp>(x, y).out();
  auto local2_w = builder.Build<MatmulOp>(local2_z, y).out();
  builder.Build<pir::TuplePushOp>(inlet_1,
                                  std::initializer_list<pir::Value>{local2_z});
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{local2_w});

  builder.SetInsertionPointToBlockEnd(block);

  // build backward network
  auto out_grad = builder.Build<FullOp>(std::vector<int64_t>{2, 2}, 1.0f).out();
  // the output of if_grad op is {x_grad, y_grad}
  auto if_grad =
      builder.Build<IfOp>(cond, std::vector<pir::Type>{x.type(), y.type()});

  // construct the true block of if_grad
  builder.SetInsertionPointToStart(&if_grad.true_block());

  auto pop_local1_z =
      builder.Build<pir::TuplePopOp>(outlet_0).outlet_element(0);
  auto local1_add_grad_op = builder.Build<AddGradOp>(pop_local1_z, y, out_grad);
  auto pop_local1_z_grad = local1_add_grad_op.x_grad(),
       local1_y_grad_0 = local1_add_grad_op.y_grad();
  auto local1_add_grad_op_1 = builder.Build<AddGradOp>(x, y, pop_local1_z_grad);
  auto local1_x_grad = local1_add_grad_op_1.x_grad(),
       local1_y_grad_1 = local1_add_grad_op_1.y_grad();
  auto local1_y_grad =
      builder.Build<AddOp>(local1_y_grad_0, local1_y_grad_1).out();

  std::string x_grad = "x_grad";
  builder.Build<pir::ShadowOutputOp>(local1_x_grad, x_grad);

  std::string y_grad = "y_grad";
  builder.Build<pir::ShadowOutputOp>(local1_y_grad, y_grad);

  std::string z_grad = "z_grad";
  builder.Build<pir::ShadowOutputOp>(pop_local1_z_grad, z_grad);

  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{local1_x_grad, local1_y_grad});

  // construct the false block of if_grad
  builder.SetInsertionPointToStart(&if_grad.false_block());
  auto pop_local2_z =
      builder.Build<pir::TuplePopOp>(outlet_1).outlet_element(0);
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

  builder.SetInsertionPointToBlockEnd(block);

  LOG(INFO) << program;

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
#if defined(PADDLE_WITH_CUDA)
  place = phi::GPUPlace();
#endif
  paddle::framework::Scope scope;

  paddle::framework::InterpreterCore test_core(
      place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({x_grad, y_grad, z_grad});

  test_core.Run({});

  auto x_grad_tensor = test_core.DebugVar(x_grad)->Get<phi::DenseTensor>();
  auto y_grad_tensor = test_core.DebugVar(y_grad)->Get<phi::DenseTensor>();
  auto z_grad_tensor = test_core.DebugVar(z_grad)->Get<phi::DenseTensor>();

  EXPECT_EQ(x_grad_tensor.data<float>()[0], 1.0);
  EXPECT_EQ(y_grad_tensor.data<float>()[0], 2.0);
  EXPECT_EQ(z_grad_tensor.data<float>()[0], 1.0);
}
