/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

std::shared_ptr<::pir::Program> BuildAllOpSupportCinnGraph() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> tan -> relu -> tan -> relu
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(full_op_x->result(0));
  auto sin_op_x = builder.Build<paddle::dialect::SinOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(sin_op_x->result(0));
  auto cos_op_y = builder.Build<paddle::dialect::CosOp>(tan_op_y->result(0));

  return program;
}

TEST(BuildCinnPassTest, AllOpSupportCinn) {
  auto origin_program = BuildAllOpSupportCinnGraph();
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(
      pm.Run(origin_program.get()),
      true,
      common::errors::InvalidArgument("Origin program not run. Expected run."));
  LOG(INFO) << "after pass: " << *origin_program;

  PADDLE_ENFORCE_EQ(origin_program->block()->size(),
                    1u,
                    common::errors::InvalidArgument(
                        "Size of block of origin program mismatch. "
                        "Expected 1 but received %d.",
                        origin_program->block()->size()));
  pir::Operation& group_op = origin_program->block()->front();
  pir::Block* group_block = group_op.dyn_cast<cinn::dialect::GroupOp>().block();
  PADDLE_ENFORCE_EQ(
      group_block->size(),
      6u,
      common::errors::InvalidArgument(
          "Size of group block mismatch. Expected 6 but received %d.",
          group_block->size()));

  std::vector<std::string> op_names = {
      paddle::dialect::FullOp::name(),
      paddle::dialect::TanOp::name(),
      paddle::dialect::SinOp::name(),
      paddle::dialect::TanOp::name(),
      paddle::dialect::CosOp::name(),
      pir::YieldOp::name(),
  };
  int index = 0;
  for (auto& op : *group_block) {
    PADDLE_ENFORCE_EQ(
        op.name(),
        op_names[index++],
        common::errors::InvalidArgument("Op name mismatch. Please check!"));
  }
}

std::shared_ptr<::pir::Program> BuildNoOpSupportCinnGraph() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // ones -> hardswish -> square -> unsqueeze
  const std::vector<int64_t> shape = {64, 128};
  const std::vector<int64_t> axis = {0};
  auto ones_op_x = builder.Build<paddle::dialect::OnesOp>(
      shape, phi::DataType::FLOAT32, phi::GPUPlace());
  auto hardswish_op_y =
      builder.Build<paddle::dialect::HardswishOp>(ones_op_x->result(0));
  auto square_op_y =
      builder.Build<paddle::dialect::SquareOp>(hardswish_op_y->result(0));
  return program;
}

TEST(BuildCinnPassTest, NoOpSupportCinn) {
  auto origin_program = BuildNoOpSupportCinnGraph();
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  PADDLE_ENFORCE_EQ(
      pm.Run(origin_program.get()),
      true,
      common::errors::InvalidArgument("Origin program not run. Expected run."));
  LOG(INFO) << "after pass: " << *origin_program;

  PADDLE_ENFORCE_EQ(origin_program->block()->size(),
                    3u,
                    common::errors::InvalidArgument(
                        "Size of block of origin program mismatch. "
                        "Expected 3 but received %d.",
                        origin_program->block()->size()));

  std::vector<std::string> op_names = {paddle::dialect::OnesOp::name(),
                                       paddle::dialect::HardswishOp::name(),
                                       paddle::dialect::SquareOp::name()};
  int index = 0;
  for (auto& op : *origin_program->block()) {
    PADDLE_ENFORCE_EQ(
        op.name(),
        op_names[index++],
        common::errors::InvalidArgument("Op name mismatch. Please check!"));
  }
}

std::shared_ptr<::pir::Program> BuildOneCinnSubgraph() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> acosh -> relu -> square -> unsqueeze
  const std::vector<int64_t> axis = {0};

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());

  auto acosh_op_x =
      builder.Build<paddle::dialect::AcoshOp>(full_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::SinOp>(acosh_op_x->result(0));
  auto square_op_y =
      builder.Build<paddle::dialect::SquareOp>(relu_op_y->result(0));
  auto unsqueeze_op_x =
      builder.Build<paddle::dialect::UnsqueezeOp>(square_op_y->result(0), axis);
  return program;
}

TEST(BuildCinnPassTest, OneCinnSubgraph) {
  auto origin_program = BuildOneCinnSubgraph();
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  PADDLE_ENFORCE_EQ(
      pm.Run(origin_program.get()),
      true,
      common::errors::InvalidArgument("Origin program not run. Expected run."));
  LOG(INFO) << "after pass: " << *origin_program;

  PADDLE_ENFORCE_EQ(origin_program->block()->size(),
                    4u,
                    common::errors::InvalidArgument(
                        "Size of block of origin program mismatch. "
                        "Expected 4 but received %d.",
                        origin_program->block()->size()));
  pir::Operation& group_op = origin_program->block()->front();
  pir::Block* group_block = group_op.dyn_cast<cinn::dialect::GroupOp>().block();
  PADDLE_ENFORCE_EQ(
      group_block->size(),
      4u,
      common::errors::InvalidArgument(
          "Size of group block mismatch. Expected 4 but received %d.",
          group_block->size()));

  std::vector<std::string> op_names = {
      paddle::dialect::FullOp::name(),
      paddle::dialect::AcoshOp::name(),
      paddle::dialect::SinOp::name(),
      pir::YieldOp::name(),
  };
  int index = 0;
  for (auto& op : *group_block) {
    PADDLE_ENFORCE_EQ(
        op.name(),
        op_names[index++],
        common::errors::InvalidArgument("Op name mismatch. Please check!"));
  }
}

std::shared_ptr<::pir::Program> BuildMultiCinnSubgraph() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> acosh -> hardswish -> square -> unsqueeze -> relu
  const std::vector<int64_t> axis = {0};

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());

  auto acosh_op_x =
      builder.Build<paddle::dialect::AcoshOp>(full_op_x->result(0));
  auto hardswish_op_y =
      builder.Build<paddle::dialect::HardswishOp>(acosh_op_x->result(0));
  auto square_op_y =
      builder.Build<paddle::dialect::SquareOp>(hardswish_op_y->result(0));
  auto unsqueeze_op_x =
      builder.Build<paddle::dialect::UnsqueezeOp>(square_op_y->result(0), axis);
  auto relu_op_y =
      builder.Build<paddle::dialect::SinOp>(unsqueeze_op_x->result(0));
  return program;
}

TEST(BuildCinnPassTest, MultiCinnSubgraph) {
  auto origin_program = BuildMultiCinnSubgraph();
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  PADDLE_ENFORCE_EQ(
      pm.Run(origin_program.get()),
      true,
      common::errors::InvalidArgument("Origin program not run. Expected run."));
  LOG(INFO) << "after pass: " << *origin_program;

  PADDLE_ENFORCE_EQ(origin_program->block()->size(),
                    5u,
                    common::errors::InvalidArgument(
                        "Size of block of origin program mismatch. "
                        "Expected 5 but received %d.",
                        origin_program->block()->size()));
  pir::Operation* group_op = &origin_program->block()->front();
  pir::Block* group_block =
      group_op->dyn_cast<cinn::dialect::GroupOp>().block();
  PADDLE_ENFORCE_EQ(
      group_block->size(),
      3u,
      common::errors::InvalidArgument(
          "Size of group block mismatch. Expected 3 but received %d.",
          group_block->size()));

  std::vector<std::string> op_names_front = {
      paddle::dialect::FullOp::name(),
      paddle::dialect::AcoshOp::name(),
      pir::YieldOp::name(),
  };
  int index = 0;
  for (auto& op : *group_block) {
    PADDLE_ENFORCE_EQ(
        op.name(),
        op_names_front[index++],
        common::errors::InvalidArgument("Op name mismatch. Please check!"));
  }

  group_op = &origin_program->block()->back();
  group_block = group_op->dyn_cast<cinn::dialect::GroupOp>().block();
  PADDLE_ENFORCE_EQ(
      group_block->size(),
      3u,
      common::errors::InvalidArgument(
          "Size of group block mismatch. Expected 3 but received %d.",
          group_block->size()));

  std::vector<std::string> op_names_back = {
      paddle::dialect::UnsqueezeOp::name(),
      paddle::dialect::SinOp::name(),
      pir::YieldOp::name(),
  };
  index = 0;
  for (auto& op : *group_block) {
    PADDLE_ENFORCE_EQ(
        op.name(),
        op_names_back[index++],
        common::errors::InvalidArgument("Op name mismatch. Please check!"));
  }
}
