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

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/attribute_base.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_attribute_storage.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_parser.h"
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/core/utils.h"

using PaddleDialect = paddle::dialect::PaddleDialect;
using AttributeStorage = ir::AttributeStorage;

enum TestType {
  AttributeTest = 0,
  TypeTest = 1,
  ProgramTest = 2,
};

class TestTask {
 public:
  TestType test_type;
  std::string test_info;

 public:
  TestTask(TestType test_type, std::string test_info) {
    this->test_info = test_info;
    this->test_type = test_type;
  }
};

class ParserTest {
 private:
  std::ifstream& test_text;

 public:
  explicit ParserTest(std::ifstream& test_text) : test_text(test_text) {}
  TestTask* GetTestTask();
  bool ConsumeTestTask(TestTask* test_task, ir::IrContext* ctx);
};

TestTask* ParserTest::GetTestTask() {
  if (test_text.peek() == EOF) {
    return nullptr;
  }
  std::string test_info;
  while (test_text.peek() != '/') {
    test_text.get();
  }
  while (test_text.peek() != ' ') {
    test_text.get();
  }
  test_text.get();
  std::string test_type_info;
  while (test_text.peek() != '\n') {
    test_type_info += test_text.get();
  }
  test_text.get();
  while (test_text.peek() != '/' && test_text.peek() != EOF) {
    test_info += test_text.get();
  }
  if (test_type_info == "attribute") {
    return new TestTask(AttributeTest, test_info);
  } else if (test_type_info == "type") {
    return new TestTask(TypeTest, test_info);
  } else if (test_type_info == "program") {
    return new TestTask(ProgramTest, test_info);
  }
  return nullptr;
}

bool ParserTest::ConsumeTestTask(TestTask* test_task, ir::IrContext* ctx) {
  std::string test_info = test_task->test_info;
  TestType test_type = test_task->test_type;
  std::unique_ptr<ir::IrPrinter> printer;
  std::unique_ptr<ir::IrParser> parser;
  std::stringstream is(test_info);
  parser.reset(new ir::IrParser(ctx, is));
  std::vector<std::string> before_parser_tokens;
  while (parser->PeekToken().token_type_ != EOF_) {
    before_parser_tokens.push_back(parser->ConsumeToken().val_);
  }
  std::stringstream is_par(test_info);
  std::stringstream os;
  if (test_type == AttributeTest) {
    auto attr = ir::Attribute::Parse(is_par, ctx);
    attr.Print(os);
  } else if (test_type == ProgramTest) {
    auto program = ir::Program::Parse(is_par, ctx);
    program->Print(os);
  } else if (test_type == TypeTest) {
    auto type = ir::Type::Parse(is_par, ctx);
    type.Print(os);
  }
  parser.reset(new ir::IrParser(ctx, os));
  std::vector<std::string> after_parser_tokens;
  while (parser->PeekToken().token_type_ != EOF_) {
    auto str = parser->ConsumeToken().val_;
    after_parser_tokens.push_back(str);
  }
  delete test_task;
  if (after_parser_tokens.size() != before_parser_tokens.size()) {
    return false;
  }

  for (size_t i = 0; i < after_parser_tokens.size(); i++) {
    if (after_parser_tokens[i] != before_parser_tokens[i]) {
      return false;
    }
  }

  return true;
}

TEST(IrParserTest, TestParserByFile) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  std::ifstream is("TestParserText.txt");
  EXPECT_TRUE(is.is_open());
  ParserTest parser_test(is);
  bool is_test = false;
  while (TestTask* test_task = parser_test.GetTestTask()) {
    is_test = true;
    bool ans = parser_test.ConsumeTestTask(test_task, ctx);
    EXPECT_TRUE(ans);
  }
  is.close();
  EXPECT_TRUE(is_test);
}
