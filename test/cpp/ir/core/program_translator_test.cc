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
#include <sstream>
#include <string>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

using PaddleDialect = paddle::dialect::PaddleDialect;
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

TEST(PaddleDialectTest, MainProgram) {
  auto p = load_from_file("resnet50_main.prog");
  EXPECT_EQ(p.Size(), 1u);

  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);

  // ops.size() = op size in BlockDesc + get_parameter_op + combine op + int
  // array op + full op (Note: p already has a full)
  EXPECT_EQ(program->block()->size(),
            p.Block(0).OpSize() + program->parameters_num() + 20 + 5 + 8);
  EXPECT_GT(ss.str().size(), 0u);
}

TEST(PaddleDialectTest, StartupProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  EXPECT_EQ(p.Size(), 1u);

  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
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
