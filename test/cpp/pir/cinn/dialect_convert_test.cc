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
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  auto sum_op =
      builder.Build<paddle::dialect::SumOp>(full_input_op.result(0),
                                            std::vector<int64_t>({-1}),
                                            phi::DataType::FLOAT32,
                                            true);
  auto relu_op = builder.Build<paddle::dialect::ReluOp>(sum_op.result(0));
  auto exp_op = builder.Build<paddle::dialect::ExpOp>(sum_op.result(0));
}

void BuildProgramMax(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  auto max_op = builder.Build<paddle::dialect::MaxOp>(
      full_input_op.result(0), std::vector<int64_t>({-1}), true);
  auto relu_op = builder.Build<paddle::dialect::ReluOp>(max_op.result(0));
  auto exp_op = builder.Build<paddle::dialect::ExpOp>(max_op.result(0));
}

TEST(DrrTest, reduce_sum) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pm.Run(&program);

  auto it = program.block()->begin();

  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::FullOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<cinn::dialect::ReduceSumOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "cinn::dialect::ReduceSumOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::ReluOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::ReluOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::ExpOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::ExpOp, but it is not."));
}

TEST(DrrTest, reduce_max) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgramMax(builder);

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pm.Run(&program);

  auto it = program.block()->begin();

  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::FullOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::FullOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<cinn::dialect::ReduceMaxOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "cinn::dialect::ReduceMaxOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::ReluOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::ReluOp, but it is not."));
  it++;
  PADDLE_ENFORCE_EQ(it->isa<paddle::dialect::ExpOp>(),
                    true,
                    common::errors::InvalidArgument(
                        "The operation should be of type "
                        "paddle::dialect::ExpOp, but it is not."));
}
