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

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
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
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

#define PROGRAM "program"
#define BASE_CODE "base_code"
#define MAGIC "magic"
#define PIRVERSION "version"
#define TRAINABLE "trainable"
#define PIR "pir"

// TEST(save_load_version_compat, op_patch_test) {
//   // (1) Init environment.
//   pir::IrContext *ctx = pir::IrContext::Instance();

//   // (2) Create an empty program object
//   pir::Program program(ctx);
//   //   pir::Program *program = new pir::Program();
//   EXPECT_EQ(program.block()->empty(), true);
//   const uint64_t pir_version = 0;
//   pir::PatchBuilder builder(pir_version);
//   builder.SetFileVersion(1);
//   // const char* paddle_root = PADDLE_ROOT;
//   // VLOG(8) << "Paddle path: " << paddle_root;
//   // std::filesystem::path patch_path =
//   //     std::filesystem::path(paddle_root) / "test" / "cpp" / "pir" /
//   //     "serialize_deserialize" / "patch";
//   std::filesystem::path patch_path("/patch")
//   VLOG(8) << "Patch path: " << patch_path;
//   builder.BuildPatch(patch_path.string(), 2, 2);
// }

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
      const char *paddle_root = PADDLE_ROOT;
      VLOG(8) << "Paddle path: " << paddle_root;
      std::filesystem::path patch_path = std::filesystem::path(paddle_root) /
                                         "test" / "cpp" / "pir" /
                                         "serialize_deserialize" / "patch";
      // std::filesystem::path patch_path("patch");
      VLOG(8) << "Patch path: " << patch_path;
      builder.BuildPatch(patch_path.string(), 2, 2);
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument("Invalid model file."));
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
// TEST(save_load_version_compat, attribute_patch_test1) {
//   pir::IrContext *ctx = pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<test::TestDialect>();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

//   pir::Program program(ctx);
//   auto block = program.block();
//   pir::Builder builder(ctx, block);

//   // Create a float32 DenseTensor Parameter and save into Program
//   pir::Type fp32_dtype = pir::Float32Type::get(ctx);

//   // Def a = ParameterOp("a")
//   std::string op1_name = pir::ParameterOp::name();
//   pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
//   std::unordered_map<std::string, pir::Attribute> op1_attribute{
//       {"parameter_name", pir::StrAttribute::get(ctx, "a")}};
//   pir::Operation *op1 =
//       pir::Operation::Create({}, op1_attribute, {fp32_dtype}, op1_info);
//   program.block()->push_back(op1);

//   // Def b = Constant("b")
//   std::string op2_name = std::string(pir::ConstantOp::name());
//   pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
//   pir::AttributeMap attr_map;
//   attr_map.insert(std::pair<std::string, pir::Attribute>(
//       "value", pir::FloatAttribute::get(ctx, 2.0)));
//   pir::Operation *op2 =
//       pir::Operation::Create({}, attr_map, {fp32_dtype}, op2_info);
//   program.block()->push_back(op2);

//   std::cout << program << std::endl;
//   // Save the program into file
//   pir::WriteModule(
//       program, "./test_save_load", /*pir_version*/ 1, true, false, true);
//   // Load the program from file
//   pir::Program new_program(ctx);
//   ReadModuleForTest("./test_save_load", &new_program, 2);
//   std::cout << new_program << std::endl;

//   EXPECT_EQ(new_program.block()->front().name(), op1->name());
//   EXPECT_EQ(new_program.block()->back().name(), op2->name());
//   // In patch yaml, the value of attribute "parameter_name" in
//   builtin.parameter is changed into "fc_0"
//   EXPECT_EQ(new_program.block()->front().attribute("parameter_name").dyn_cast<::pir::StrAttribute>().AsString(),
//   "fc_0");
//   // In patch yaml, the value of attribute "stop_gradient" in
//   builtin.parameter is changed into false std::cout <<
//   new_program.block()->front().attribute("stop_gradient").dyn_cast<::pir::ArrayAttribute>().AsVector()[0]
//   << std::endl;
//   EXPECT_EQ(new_program.block()->front().attribute("stop_gradient").dyn_cast<::pir::ArrayAttribute>().AsVector()[0].dyn_cast<::pir::BoolAttribute>().data(),
//   false);
//   // In patch yaml, the value of attribute "value" in builtin.constant is
//   changed into 1.0
//   // Also, the type of attribute "FloatAttribute" is changed into
//   "DoubleAttribute".
//   EXPECT_EQ(new_program.block()->back().attribute("value").dyn_cast<::pir::DoubleAttribute>().data(),
//   1.0);
//   // In patch yaml, the type of Float32Type is changed into Float64Type
//   EXPECT_EQ(new_program.block()->front().result(0).type(),
//   pir::Float64Type::get(ctx));
//   EXPECT_EQ(new_program.block()->back().result(0).type(),
//   pir::Float64Type::get(ctx));
// }
// namespace test {
// class AddOp : public pir::Op<AddOp> {
//  public:
//   using Op::Op;
//   static const char *name() { return "test.add"; }
//   static constexpr const char **attributes_name = nullptr;
//   static constexpr uint32_t attributes_num = 0;
//   void VerifySig();
//   static void Build(pir::Builder &builder,             // NOLINT
//                     pir::OperationArgument &argument,  // NOLINT
//                     pir::Value l_operand,
//                     pir::Value r_operand,
//                     pir::Type sum_type);
// };
// void AddOp::VerifySig() {
//   if (num_operands() != 2) {
//     PADDLE_THROW(
//         common::errors::Fatal("The size of inputs must be equal to 2."));
//   }
//   if (num_results() != 1) {
//     PADDLE_THROW(
//         common::errors::Fatal("The size of outputs must be equal to 1."));
//   }
// }
// void AddOp::Build(pir::Builder &,
//                   pir::OperationArgument &argument,
//                   pir::Value l_operand,
//                   pir::Value r_operand,
//                   pir::Type sum_type) {
//   argument.AddInput(l_operand);
//   argument.AddInput(r_operand);
//   argument.AddOutput(sum_type);
// }
// }
// IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::AddOp)
// IR_DEFINE_EXPLICIT_TYPE_ID(test::AddOp)

TEST(save_load_version_compat, op_patch_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);
  // phi::DDim dims = {2, 2};
  // phi::DataLayout data_layout = phi::DataLayout::NCHW;
  // phi::LoD lod = {{0, 1, 2}};
  // size_t offset = 0;

  // pir::DenseTensorType dense_tensor_dtype =
  //     pir::DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset);

  // pir::Operation *op1 = builder.Build<test::Operation1>();
  // pir::Operation *op2 = builder.Build<test::Operation2>();

  // (2) Get registered operations.
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"op1_attr1", pir::StrAttribute::get(ctx, "op1_attr1")},
      {"op1_attr2", pir::StrAttribute::get(ctx, "op1_attr2")}};

  pir::Operation *op1 = pir::Operation::Create(
      {}, op1_attribute, {pir::Float32Type::get(ctx)}, op1_info);
  block->push_back(op1);
  pir::Operation *op2 = pir::Operation::Create(
      {}, op1_attribute, {pir::Float32Type::get(ctx)}, op2_info);
  block->push_back(op2);
  std::string op3_name = std::string(paddle::dialect::AddOp::name());
  pir::OpInfo op3_info = ctx->GetRegisteredOpInfo(op3_name);
  // pir::AttributeMap attr_map;
  // attr_map.insert(std::pair<std::string, pir::Attribute>(
  //     "value", pir::FloatAttribute::get(ctx, 2.0)));
  std::vector<pir::Value> inputs;
  inputs.push_back(op1->result(0));
  inputs.push_back(op2->result(0));
  pir::Operation *op3 = pir::Operation::Create(
      inputs, {}, {pir::Float32Type::get(ctx)}, op3_info);
  program.block()->push_back(op3);
  std::cout << "op3: " << op3->result(0).type() << std::endl;
  std::cout << program << std::endl;
  pir::WriteModule(
      program, "./test_save_load", /*pir_version*/ 1, true, false, true);
}
