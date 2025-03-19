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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

std::vector<pir::Type> CreateDenseTensorTypes(const phi::DDim &dims) {
  pir::IrContext *ctx = ::pir::IrContext::Instance();
  pir::Type fp32_dtype = ::pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {};
  size_t offset = 0;
  std::vector<::pir::Type> op_output_types = {::pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset)};
  return op_output_types;
}

void BuildProgram(pir::Builder &builder) {  // NOLINT
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim({4, 3, 16})));
  builder.SetInsertionPointToBlockEnd(group_op.block());
  paddle::dialect::FullOp full_input_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_input_y = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{16}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());
  auto add_op = builder.Build<paddle::dialect::AddOp>(full_input_x.result(0),
                                                      full_input_y.result(0));

  auto relu_op = builder.Build<paddle::dialect::ReluOp>(add_op.result(0));
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{relu_op.out()});
}

void BuildProgramBoth(pir::Builder &builder) {  // NOLINT
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim({10, 10})));
  builder.SetInsertionPointToBlockEnd(group_op.block());
  paddle::dialect::FullOp full_input_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{10, 1},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_input_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 10},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  auto add_op = builder.Build<paddle::dialect::AddOp>(full_input_x.result(0),
                                                      full_input_y.result(0));

  auto relu_op = builder.Build<paddle::dialect::ReluOp>(add_op.result(0));
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{relu_op.out()});
}

void BuildProgramSubBoth(pir::Builder &builder) {  // NOLINT
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim({10, 10})));
  builder.SetInsertionPointToBlockEnd(group_op.block());
  paddle::dialect::FullOp full_input_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{10, 1},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_input_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 10},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  auto sub_op = builder.Build<paddle::dialect::SubtractOp>(
      full_input_x.result(0), full_input_y.result(0));

  auto relu_op = builder.Build<paddle::dialect::ReluOp>(sub_op.result(0));
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{relu_op.out()});
}

TEST(PatternRewrite, broadcast_elementwise) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());

  pm.Run(&program);

  auto it = program.block()
                ->begin()
                ->dyn_cast<cinn::dialect::GroupOp>()
                .block()
                ->begin();

  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::AddOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected AddOp but found different operation type: " +
                        std::string(it->name())));
}

TEST(PatternRewrite, broadcast_elementwise_both) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgramBoth(builder);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());

  pm.Run(&program);

  auto it = program.block()
                ->begin()
                ->dyn_cast<cinn::dialect::GroupOp>()
                .block()
                ->begin();

  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::AddOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected AddOp but found different operation type: " +
                        std::string(it->name())));
}

TEST(PatternRewrite, broadcast_elementwise_sub_both) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgramSubBoth(builder);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());

  pm.Run(&program);

  auto it = program.block()
                ->begin()
                ->dyn_cast<cinn::dialect::GroupOp>()
                .block()
                ->begin();

  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Expected FullOp but found different operation type: " +
                        std::string(it->name())));
  it++;
  PADDLE_ENFORCE_EQ(
      it->isa<paddle::dialect::SubtractOp>(),
      true,
      common::errors::PreconditionNotMet(
          "Expected SubtractOp but found different operation type: " +
          std::string(it->name())));
}
