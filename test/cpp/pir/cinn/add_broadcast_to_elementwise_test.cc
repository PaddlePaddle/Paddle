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
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
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
}

void BuildProgramBoth(pir::Builder &builder) {  // NOLINT
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
}

void BuildProgramSubBoth(pir::Builder &builder) {  // NOLINT
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
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());

  pm.Run(&program);

  auto it = program.block()->begin();

  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::AddOp>(), true);
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
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());

  pm.Run(&program);

  auto it = program.block()->begin();

  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::AddOp>(), true);
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
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());

  pm.Run(&program);

  auto it = program.block()->begin();

  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::FullOp>(), true);
  it++;
  CHECK_EQ(it->isa<paddle::dialect::SubtractOp>(), true);
}
