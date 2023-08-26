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

#include "paddle/ir/core/ir_parser.h"
#include <gtest/gtest.h>
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
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/core/utils.h"

using PaddleDialect = paddle::dialect::PaddleDialect;
using AttributeStorage = ir::AttributeStorage;
using string = std::string;
using std::ifstream;
using std::stringstream;

enum TestType {
  AttributeMapTest = 0,
  TypeTest = 1,
  BlockTest = 2,
};

class TestTask {
 public:
  TestType test_type;
  string test_info;

 public:
  TestTask(TestType test_type, string test_info) {
    this->test_info = test_info;
    this->test_type = test_type;
  }
};

class ParserTest {
 private:
  ifstream& test_text;

 public:
  explicit ParserTest(ifstream& test_text) : test_text(test_text) {}
  TestTask* GetTestTask();
  bool ConsumeTestTask(TestTask* test_task, ir::IrContext* ctx);
};

TestTask* ParserTest::GetTestTask() {
  if (test_text.peek() == EOF) {
    return nullptr;
  }
  string test_info;
  while (test_text.peek() != '/') {
    test_text.get();
  }
  while (test_text.peek() != ' ') {
    test_text.get();
  }
  test_text.get();
  string test_type_info;
  while (test_text.peek() != '/n') {
    test_type_info += test_text.get();
  }
  test_text.get();
  while (test_text.peek() != '/' || test_text.peek() != EOF) {
    test_info += test_text.get();
  }
  if (test_type_info == "attributemap") {
    return new TestTask(AttributeMapTest, test_info);
  } else if (test_type_info == "type") {
    return new TestTask(TypeTest, test_info);
  } else if (test_type_info == "block") {
    return new TestTask(BlockTest, test_info);
  }
  return nullptr;
}

bool ParserTest::ConsumeTestTask(TestTask* test_task, ir::IrContext* ctx) {
  string test_info = test_task->test_info;
  TestType test_type = test_task->test_type;
  ir::IrPrinter* printer;
  ir::IrParser* parser;
  stringstream is(test_info);
  parser = new ir::IrParser(ctx, is);
  std::vector<string> before_parser_tokens;
  while (parser->PeekToken().token_type_ != EOF_) {
    before_parser_tokens.push_back(parser->GetToken().val_);
  }
  parser = new ir::IrParser(ctx, is);
  stringstream os;
  if (test_type == AttributeMapTest) {
    AttributeMap attributes = parser->ParseAttributeMap();
    printer = new ir::IrPrinter(os);
    std::map<std::string, ir::Attribute, std::less<std::string>>
        order_attributes(attributes.begin(), attributes.end());
    os << " {";
    PrintInterleave(
        order_attributes.begin(),
        order_attributes.end(),
        [printer](std::pair<std::string, ir::Attribute> it) {
          printer->os << it.first;
          printer->os << ":";
          printer->PrintAttribute(it.second);
        },
        [printer]() { printer->os << ","; });

    os << "}";
  } else if (test_type == BlockTest) {
    ir::Block* block = new ir::Block();
    parser->ParseBlock(*block);
    printer = new ir::IrPrinter(os);
    printer->PrintBlock(block);
    parser = new ir::IrParser(ctx, os);
    std::vector<string> after_parser_tokens;
    while (parser->PeekToken().token_type_ != EOF_) {
      after_parser_tokens.push_back(parser->GetToken().val_);
    }
    delete printer;
    delete parser;
    delete test_task;
    if (after_parser_tokens.size() != before_parser_tokens.size()) return false;

    for (int i = 0; i < after_parser_tokens.size(); i++) {
      if (after_parser_tokens[i] != before_parser_tokens[i]) return false;
    }

    return true;
  } else if (test_type == TypeTest) {
    ir::Type type = parser->ParseType();
    printer = new ir::IrPrinter(os);
    printer->PrintType(type);
  }
  parser = new ir::IrParser(ctx, os);
  std::vector<string> after_parser_tokens;
  while (parser->PeekToken().token_type_ != EOF_) {
    after_parser_tokens.push_back(parser->GetToken().val_);
  }
  delete printer;
  delete parser;
  delete test_task;
  if (after_parser_tokens.size() != before_parser_tokens.size()) return false;

  for (int i = 0; i < after_parser_tokens.size(); i++) {
    if (after_parser_tokens[i] != before_parser_tokens[i]) return false;
  }

  return true;
}

TEST(IrParserTest, TestParserByFile) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  std::ifstream is("TestParserText.txt");
  ParserTest parser_test(is);
  bool test_ans = true;
  TestTask* test_task;
  while (test_task = parser_test.GetTestTask()) {
    test_ans &= parser_test.ConsumeTestTask(test_task, ctx);
  }
  EXPECT_TRUE(test_ans);
}
