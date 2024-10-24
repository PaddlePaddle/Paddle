// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"

TEST(SaveTest, uncompelteted_parameter) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  paddle::dialect::FullOp full_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(pir::ParameterOp::name());
  std::vector<pir::Value> inputs;
  std::vector<pir::Type> output_types;
  output_types.push_back(full_op1.out().type());

  pir::AttributeMap attributes;
  attributes.insert(
      {"parameter_name", pir::StrAttribute::get(ctx, "test_param")});
  pir::Operation* op =
      pir::Operation::Create(inputs, attributes, output_types, op_info);
  program.block()->push_back(op);

  pir::WriteModule(
      program, "./test_param", /*pir_version*/ 0, true, false, true);

  pir::Program new_program(ctx);
  pir::ReadModule("./test_param", &new_program, /*pir_version*/ 0);

  pir::Operation& new_op = new_program.block()->back();
  EXPECT_EQ(new_op.attribute("is_distributed").isa<pir::ArrayAttribute>(),
            true);
  EXPECT_EQ(new_op.attribute("is_parameter").isa<pir::ArrayAttribute>(), true);
  EXPECT_EQ(new_op.attribute("need_clip").isa<pir::ArrayAttribute>(), true);
  EXPECT_EQ(new_op.attribute("parameter_name").isa<pir::StrAttribute>(), true);
  EXPECT_EQ(new_op.attribute("persistable").isa<pir::ArrayAttribute>(), true);
  EXPECT_EQ(new_op.attribute("stop_gradient").isa<pir::ArrayAttribute>(), true);
  EXPECT_EQ(new_op.attribute("trainable").isa<pir::ArrayAttribute>(), true);
}
