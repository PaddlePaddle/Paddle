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
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/type_id.h"
#include "test/cpp/pir/custom_engine/custom_engine_op.h"

TEST(op_test, region_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Dialect *custom_engine_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomEngineDialect>();
  EXPECT_EQ(custom_engine_dialect != nullptr, true);
  ctx->RegisterOpInfo(custom_engine_dialect,
                      pir::TypeId::get<paddle::dialect::FakeEngineOp>(),
                      paddle::dialect::FakeEngineOp::name(),
                      paddle::dialect::FakeEngineOp::interface_set(),
                      paddle::dialect::FakeEngineOp::GetTraitSet(),
                      paddle::dialect::FakeEngineOp::attributes_num,
                      paddle::dialect::FakeEngineOp::attributes_name,
                      paddle::dialect::FakeEngineOp::VerifySigInvariants,
                      paddle::dialect::FakeEngineOp::VerifyRegionInvariants);

  pir::Program program(ctx);
  pir::Block *block = program.block();
  pir::Builder builder(ctx, block);

  auto full_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2}, 100);
  auto full_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2}, 10);

  auto buildin_combine_op = builder.Build<pir::CombineOp>(
      std::vector<pir::Value>{full_op1.result(0), full_op2.result(0)});

  pir::OpInfo fake_engine_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::FakeEngineOp::name());

  std::vector<pir::Type> out_types;
  out_types.push_back(
      pir::DenseTensorType::get(pir::IrContext::Instance(),
                                pir::Float32Type::get(ctx),
                                phi::DDim(std::vector<int64_t>{2, 2}.data(), 2),
                                phi::DataLayout::kNCHW,
                                phi::LoD(),
                                0));
  pir::Type out_vector_type =
      pir::VectorType::get(pir::IrContext::Instance(), out_types);
  std::vector<pir::Type> output_types = {out_vector_type};

  pir::AttributeMap attribute_map;
  std::vector<pir::Attribute> val;
  val.push_back(pir::StrAttribute::get(ctx, "input_0"));
  val.push_back(pir::StrAttribute::get(ctx, "input_1"));
  attribute_map.insert({"input_names", pir::ArrayAttribute::get(ctx, val)});
  std::vector<pir::Attribute> out_val;
  out_val.push_back(pir::StrAttribute::get(ctx, "output_0"));
  out_val.push_back(pir::StrAttribute::get(ctx, "output_1"));
  attribute_map.insert(
      {"output_names", pir::ArrayAttribute::get(ctx, out_val)});

  pir::Operation *op1 = pir::Operation::Create({buildin_combine_op.result(0)},
                                               attribute_map,
                                               output_types,
                                               fake_engine_op_info);

  // (3) Test custom operation printer
  std::stringstream ss1;
  op1->Print(ss1);

  builder.Insert(op1);

  auto op2 = builder.Build<paddle::dialect::FakeEngineOp>(
      buildin_combine_op.result(0),
      std::vector<std::string>{"input_0", "input_1"},
      std::vector<std::string>{"output_0"},
      std::vector<std::vector<int64_t>>{{2, 2}},
      std::vector<phi::DataType>{phi::DataType::FLOAT32});

  std::stringstream ss2;
  op2->Print(ss2);

  EXPECT_EQ(block->size(), 5u);
}
