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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/sub_graph/sub_graph_checker.h"
#include "paddle/pir/include/core/ir_context.h"

using namespace paddle::test;  // NOLINT

std::shared_ptr<::pir::Program> BuildBasicProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> softmax(max -> subtract -> exp -> sum -> divide)
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {128, 12, 128, 128};
  std::string input_name = std::string(kInputPrefix) + "0";
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   input_name, shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto out = builder.Build<paddle::dialect::SoftmaxOp>(x, -1).result(0);

  return program;
}

std::shared_ptr<::pir::Program> BuildPrimProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  std::string input_name = std::string(kInputPrefix) + "0";

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {128, 12, 128, 128};

  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   input_name, shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  //   auto out = builder.Build<paddle::dialect::SinOp>(x).result(0);
  auto max =
      builder.Build<paddle::dialect::MaxOp>(x, std::vector<int64_t>{-1}, true)
          .result(0);
  auto sub = builder.Build<paddle::dialect::SubtractOp>(x, max).result(0);
  auto exp = builder.Build<paddle::dialect::ExpOp>(sub).result(0);
  auto sum =
      builder
          .Build<paddle::dialect::SumOp>(
              exp, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto out = builder.Build<paddle::dialect::DivideOp>(exp, sum).result(0);

  return program;
}

std::shared_ptr<::pir::Program> BuildDropOutPrimProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  std::string input_name = std::string(kInputPrefix) + "0";

  auto x =
      builder
          .Build<paddle::dialect::DataOp>(input_name,
                                          std::vector<int64_t>({128, 128, 768}),
                                          phi::DataType::FLOAT32,
                                          phi::GPUPlace())
          .result(0);

  auto prob = builder
                  .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
                                                  0.5,
                                                  phi::DataType::FLOAT32,
                                                  phi::GPUPlace())
                  .result(0);

  auto random = builder
                    .Build<paddle::dialect::UniformOp>(
                        std::vector<int64_t>({128, 128, 768}),
                        phi::DataType::FLOAT32,
                        0.0,
                        1.0,
                        0,
                        phi::GPUPlace())
                    .result(0);

  auto mask =
      builder.Build<paddle::dialect::GreaterThanOp>(random, prob).result(0);
  auto mask1 =
      builder.Build<paddle::dialect::CastOp>(mask, phi::DataType::FLOAT32)
          .result(0);
  auto mul = builder.Build<paddle::dialect::MultiplyOp>(x, mask1).result(0);
  auto neg_prob = prob =
      builder
          .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
                                          0.5,
                                          phi::DataType::FLOAT32,
                                          phi::GPUPlace())
          .result(0);
  auto out = builder.Build<paddle::dialect::DivideOp>(mul, neg_prob).result(0);

  return program;
}

std::shared_ptr<::pir::Program> BuildDropOutPhiProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  std::string input_name = std::string(kInputPrefix) + "0";

  auto x =
      builder
          .Build<paddle::dialect::DataOp>(input_name,
                                          std::vector<int64_t>({128, 128, 768}),
                                          phi::DataType::FLOAT32,
                                          phi::GPUPlace())
          .result(0);

  auto out = builder
                 .Build<paddle::dialect::DropoutOp>(
                     x, pir::Value(), 0.5, false, "upscale_in_train", 0, false)
                 .result(0);
  return program;
}

TEST(sub_graph_checker, test_softmax) {
  auto basic_program = BuildBasicProgram();
  auto prim_program = BuildPrimProgram();

  paddle::test::SubGraphChecker sub_graph_checker(basic_program, prim_program);

  ASSERT_TRUE(sub_graph_checker.CheckResult());
  std::vector<double> speed_data = sub_graph_checker.CheckSpeed();
  ASSERT_EQ(speed_data.size(), 2u);
}

TEST(sub_graph_checker, test_dropout) {
  auto basic_program = BuildDropOutPhiProgram();
  auto prim_program = BuildDropOutPrimProgram();

  paddle::test::SubGraphChecker sub_graph_checker(basic_program, prim_program);

  ASSERT_TRUE(sub_graph_checker.CheckResult());
  std::vector<double> speed_data = sub_graph_checker.CheckSpeed();
  ASSERT_EQ(speed_data.size(), 2u);
}
