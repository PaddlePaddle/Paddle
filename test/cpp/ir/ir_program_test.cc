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

#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/program.h"
#include "paddle/ir/utils.h"

class AddOp : public ir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "Add"; }
  static const char *attributes_name_[];
  static uint32_t attributes_num() { return 0; }
};
const char *AddOp::attributes_name_[] = {};

TEST(program_test, program) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  builtin_dialect->RegisterOp<AddOp>();

  // (2) Create an empty program object
  ir::Program *program = new ir::Program();
  EXPECT_EQ(program->ops().size() == 0, true);
  EXPECT_EQ(program->parameters().size() == 0, true);

  // (3) Def a program:
  // a = GetParameterOp("a")
  std::string op1_name(ir::GetParameterOp::name());
  ir::OpInfoImpl *op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::map<ir::StrAttribute, ir::Attribute> op1_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "a")}};
  ir::DictionaryAttribute op1_attribute =
      ir::DictionaryAttribute::get(ctx, op1_attribute_map);
  std::cout << "before create:" << op1_info << std::endl;
  ir::Operation *op1 = ir::Operation::create(
      {}, {ir::Float32Type::get(ctx)}, op1_attribute, op1_info, program);

  // b = GetParameterOp("b")
  std::string op2_name(ir::GetParameterOp::name());
  ir::OpInfoImpl *op2_info = ctx->GetRegisteredOpInfo(op2_name);
  std::map<ir::StrAttribute, ir::Attribute> op2_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "b")}};
  ir::DictionaryAttribute op2_attribute =
      ir::DictionaryAttribute::get(ctx, op2_attribute_map);
  ir::Operation *op2 = ir::Operation::create(
      {}, {ir::Float32Type::get(ctx)}, op2_attribute, op2_info, program);

  // c = AddOp(a, b)
  std::string op3_name(AddOp::name());
  ir::OpInfoImpl *op3_info = ctx->GetRegisteredOpInfo(op3_name);
  ir::Operation *op3 = ir::Operation::create(
      {op1->GetResultByIndex(0), op2->GetResultByIndex(0)},
      {ir::Float32Type::get(ctx)},
      nullptr,
      op3_info,
      program);

  // SetParameterOp(c, "c")
  std::string op4_name(ir::SetParameterOp::name());
  ir::OpInfoImpl *op4_info = ctx->GetRegisteredOpInfo(op4_name);
  std::map<ir::StrAttribute, ir::Attribute> op4_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "c")}};
  ir::DictionaryAttribute op4_attribute =
      ir::DictionaryAttribute::get(ctx, op4_attribute_map);
  ir::Operation *op4 = ir::Operation::create(
      {op3->GetResultByIndex(0)}, {}, op4_attribute, op4_info, program);
  std::cout << op4 << std::endl;

  // (4) Traverse Program
  std::list<ir::Operation *> ops = program->ops();
  EXPECT_EQ(ops.size() == 4, true);

  delete program;
}
