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
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_deserialize.h"
#include "paddle/fluid/pir/serialize_deserialize/include/version_compat.h"
#include "paddle/phi/common/port.h"

#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/utils.h"
#include "test/cpp/pir/tools/test1_dialect.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

#define PROGRAM "program"
#define BASE_CODE "base_code"
#define MAGIC "magic"
#define PIRVERSION "version"
#define TRAINABLE "trainable"
#define PIR "pir"

// Test for building patches.
TEST(save_load_version_compat, op_patch_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  //   pir::Program *program = new pir::Program();
  EXPECT_EQ(program.block()->empty(), true);
  const uint64_t pir_version = 0;
  pir::PatchBuilder builder(pir_version);
  builder.SetFileVersion(1);
  std::filesystem::path patch_path("patch");
  VLOG(8) << "Patch path: " << patch_path;
  builder.BuildPatch(2, 2, patch_path.string());
}

bool ReadModuleForTest(const std::string &file_path,
                       pir::Program *program,
                       uint64_t pir_version) {
  std::ifstream f(file_path);
  Json data = Json::parse(f);
  pir::PatchBuilder builder(pir_version);

  if (data.contains(BASE_CODE) && data[BASE_CODE].contains(MAGIC) &&
      data[BASE_CODE][MAGIC] == PIR) {
    uint64_t file_version =
        data.at(BASE_CODE).at(PIRVERSION).template get<uint64_t>();
    if (file_version != pir_version) {
      builder.SetFileVersion(file_version);
      std::filesystem::path patch_path("patch");
      VLOG(8) << "Patch path: " << patch_path;
      builder.BuildPatch(2, 2, patch_path.string());
    }
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("Invalid model file: %s.",
                                                   file_path));
  }

  pir::ProgramReader reader(pir_version);
  reader.RecoverProgram(&(data[PROGRAM]), program, &builder);

  if (data[BASE_CODE].contains(TRAINABLE)) {
    return data[BASE_CODE][TRAINABLE].get<bool>();
  } else {
    return false;
  }
}

// Test for attribute patch and op attribute modification.
TEST(save_load_version_compat, attribute_patch_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  // Create a float32 DenseTensor Parameter and save into Program
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);

  // Def a = ParameterOp("a")
  std::string op1_name = pir::ParameterOp::name();
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"parameter_name", pir::StrAttribute::get(ctx, "a")}};
  pir::Operation *op1 =
      pir::Operation::Create({}, op1_attribute, {fp32_dtype}, op1_info);
  program.block()->push_back(op1);

  // Def b = Constant("b")
  std::string op2_name = std::string(pir::ConstantOp::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  pir::AttributeMap attr_map;
  attr_map.insert(std::pair<std::string, pir::Attribute>(
      "value", pir::FloatAttribute::get(ctx, 2.0)));
  pir::Operation *op2 =
      pir::Operation::Create({}, attr_map, {fp32_dtype}, op2_info);
  program.block()->push_back(op2);

  // Save the program into file
  pir::WriteModule(
      program, "./test_save_load", /*pir_version*/ 1, true, false, true);
  // Load the program from file
  pir::Program new_program(ctx);
  ReadModuleForTest("./test_save_load", &new_program, 2);

  EXPECT_EQ(new_program.block()->front().name(), op1->name());
  EXPECT_EQ(new_program.block()->back().name(), op2->name());
  // In patch yaml, the value of attribute "parameter_name" in builtin.parameter
  // is changed into "fc_0"
  EXPECT_EQ(new_program.block()
                ->front()
                .attribute("parameter_name")
                .dyn_cast<::pir::StrAttribute>()
                .AsString(),
            "fc_0");
  // In patch yaml, the value of attribute "stop_gradient" in builtin.parameter
  // is changed into false
  EXPECT_EQ(new_program.block()
                ->front()
                .attribute("stop_gradient")
                .dyn_cast<::pir::ArrayAttribute>()
                .AsVector()[0]
                .dyn_cast<::pir::BoolAttribute>()
                .data(),
            false);
  // In patch yaml, the value of attribute "value" in builtin.constant is
  // changed into 1.0 Also, the type of attribute "FloatAttribute" is changed
  // into "DoubleAttribute".
  EXPECT_EQ(new_program.block()
                ->back()
                .attribute("value")
                .dyn_cast<::pir::DoubleAttribute>()
                .data(),
            1.0);
  // In patch yaml, the type of Float32Type is changed into Float64Type
  EXPECT_EQ(new_program.block()->front().result(0).type(),
            pir::Float64Type::get(ctx));
  EXPECT_EQ(new_program.block()->back().result(0).type(),
            pir::Float64Type::get(ctx));
}

// Test for op I/O and op attribute modification.
TEST(save_load_version_compat, op_patch_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<test1::Test1Dialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);

  // Get registered operations.
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"op1_attr1", pir::StrAttribute::get(ctx, "op1_attr1")},
      {"op1_attr2", pir::StrAttribute::get(ctx, "op1_attr2")}};

  pir::Operation *op1 =
      pir::Operation::Create({}, op1_attribute, {dtype}, op1_info);
  block->push_back(op1);

  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation2::name());
  pir::Operation *op2 = pir::Operation::Create({}, {}, {}, op2_info);
  program.block()->push_back(op2);

  pir::WriteModule(
      program, "./test_save_load", /*pir_version*/ 1, true, false, true);
  // Load the program from file
  pir::Program new_program(ctx);
  ReadModuleForTest("./test_save_load", &new_program, 2);

  // In patch yaml, the type of Float32Type is changed into Float64Type
  EXPECT_EQ(new_program.block()->front().result(0).type(),
            pir::Float64Type::get(ctx));
  // In patch yaml, the value of attribute "op1_attr1" in test.operation1 is
  // changed into "op1_attr1_value".
  EXPECT_EQ(new_program.block()
                ->front()
                .attribute("op1_attr1")
                .dyn_cast<::pir::StrAttribute>()
                .AsString(),
            "op1_attr1_value");
  // In patch yaml, the attribute "op1_attr2" is deleted from test.operation1.
  EXPECT_EQ(new_program.block()->front().HasAttribute("op1_attr2"), false);
  // In patch yaml, a new attribute "op1_attr3" is added into test.operation1.
  EXPECT_EQ(new_program.block()
                ->front()
                .attribute("op1_attr3")
                .dyn_cast<::pir::StrAttribute>()
                .AsString(),
            "op1_attr3_value");
  // In patch yaml, an output has been added to test.operation2, so the number
  // of results is not 0.
  EXPECT_EQ(new_program.block()->back().results().empty(), false);
}

// Test for the combination of op I/O and op_pair patch for deleting value.
TEST(save_load_version_compat, op_patch_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<test1::Test1Dialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);

  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test1::Operation4::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test1::Operation3::name());

  pir::Operation *op1 =
      pir::Operation::Create({}, {}, {dtype, dtype}, op1_info);
  block->push_back(op1);

  pir::Operation *op2 = pir::Operation::Create(
      {op1->result(0), op1->result(1)}, {}, {dtype}, op2_info);
  block->push_back(op2);

  pir::WriteModule(
      program, "./test_save_load", /*pir_version*/ 1, true, false, true);
  // Load the program from file
  pir::Program new_program(ctx);
  ReadModuleForTest("./test_save_load", &new_program, 2);

  // After applying the op_pair patch which deleted the output of
  // test1.operation4 and the input of test1.operation3 the number of results of
  // test1.operation4 is changed into 1.
  EXPECT_EQ(new_program.block()->front().num_results(), (uint64_t)1);
  // And with the combination of the op_patch which deleted another input of
  // test1.operation3, the number of operands of test1.operation3 is changed
  // into 0.
  EXPECT_EQ(new_program.block()->back().operands().empty(), true);
}

// Test for op_pair patch for adding value.
TEST(save_load_version_compat, op_patch_test3) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<test1::Test1Dialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);

  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test1::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test1::Operation2::name());
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"op1_attr1", pir::StrAttribute::get(ctx, "op1_attr1")},
      {"op1_attr3", pir::StrAttribute::get(ctx, "op1_attr3")}};

  pir::Operation *op1 =
      pir::Operation::Create({}, op1_attribute, {dtype}, op1_info);
  block->push_back(op1);

  pir::Operation *op2 = pir::Operation::Create({}, {}, {dtype}, op2_info);
  block->push_back(op2);

  pir::WriteModule(
      program, "./test_save_load", /*pir_version*/ 1, true, false, true);
  // Load the program from file
  pir::Program new_program(ctx);
  ReadModuleForTest("./test_save_load", &new_program, 2);

  // After applying the op_pair patch which deleted the output of
  // test1.operation4 and the input of test1.operation3 the number of results of
  // test1.operation4 is changed into 1.
  EXPECT_EQ(new_program.block()->front().num_results(), (uint64_t)2);
  // And with the combination of the op_patch which deleted another input of
  // test1.operation3, the number of operands of test1.operation3 is changed
  // into 0.
  EXPECT_EQ(new_program.block()->back().num_operands(), (uint64_t)1);
}
