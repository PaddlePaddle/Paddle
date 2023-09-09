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
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/ir_parser.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/program.h"

using OperatorDialect = paddle::dialect::OperatorDialect;
using ProgramDesc = paddle::framework::ProgramDesc;
using BlockDesc = paddle::framework::BlockDesc;
using OpDesc = paddle::framework::OpDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;

ProgramDesc load_from_file(const std::string &file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);

  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  return ProgramDesc(buffer);
}

TEST(OperatorDialectTest, MainProgram) {
  auto p = load_from_file("resnet50_main.prog");
  EXPECT_EQ(p.Size(), 1u);

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);

  // ops.size() = op size in BlockDesc + get_parameter_op + combine op + int
  // array op + full op (Note: p already has a full)
  EXPECT_EQ(program->block()->size(),
            p.Block(0).OpSize() + program->parameters_num() + 20 + 5 + 8);
  EXPECT_GT(ss.str().size(), 0u);
}

TEST(OperatorDialectTest, StartupProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  EXPECT_EQ(p.Size(), 1u);

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  size_t op_size = program->block()->size();
  // ops.size() = op size in BlockDesc + get_parameter_op +
  // consant_op_for_uniform
  // + consant_op for guassian
  EXPECT_EQ(op_size, p.Block(0).OpSize() + program->parameters_num() + 3 + 53);

  std::stringstream ss;
  program->Print(ss);
  EXPECT_GT(ss.str().size(), 0u);
}

TEST(RegisterInfoTest, MainProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  pir::IrContext *ctx = pir::IrContext::Instance();

  auto unregistered_ops =
      paddle::translator::CheckUnregisteredOperation(ctx, p);
  EXPECT_EQ(unregistered_ops.size(), 0u);

  auto new_op = std::unique_ptr<OpDesc>(
      new OpDesc("something must not be registered", {}, {}, {}));
  auto *block = p.MutableBlock(0);
  block->AppendAllocatedOp(std::move(new_op));

  unregistered_ops = paddle::translator::CheckUnregisteredOperation(ctx, p);
  EXPECT_EQ(unregistered_ops.size(), 1u);
  EXPECT_EQ(unregistered_ops[0], "something must not be registered");
}

TEST(IrParserTest, MainProgram) {
  auto p = load_from_file("resnet50_main.prog");
  EXPECT_EQ(p.Size(), 1u);
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);
  std::unique_ptr<pir::Program> parser_program = pir::Program::Parse(ss, ctx);
  std::stringstream ssp;
  parser_program->Print(ssp);

  EXPECT_TRUE(ssp.str() == ss.str());
}

TEST(IrParserTest, StartupProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  EXPECT_EQ(p.Size(), 1u);
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);
  std::unique_ptr<pir::Program> parser_program = pir::Program::Parse(ss, ctx);
  std::stringstream ssp;
  parser_program->Print(ssp);

  EXPECT_TRUE(ssp.str() == ss.str());
}
