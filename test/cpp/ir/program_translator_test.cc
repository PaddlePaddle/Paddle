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
#include <string>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/paddle_dialect/paddle_dialect.h"
#include "paddle/fluid/translator/translate.h"
#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/program.h"

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

TEST(PaddleDialectTest, Translator) {
  auto p = load_from_file("restnet50_main.prog");
  std::cout << p.Size() << std::endl;

  EXPECT_EQ(p.Size(), 1u);

  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto program = paddle::fluid::TranslateLegacyProgramToProgram(p);

  std::list<ir::Operation *> ops = program->ops();
  std::cout << ops.size() << std::endl;
  EXPECT_EQ(ops.size(), p.Block(0).OpSize() + program->parameters().size());
  std::cout << program.get() << (*program).ops().size() << std::endl;
  for (auto *op : (*program).ops()) {
    std::cout << op->op_name() << std::endl;
  }
  std::cout << *program << std::endl;
}
