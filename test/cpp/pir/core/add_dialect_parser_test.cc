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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/attribute_base.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_attribute_storage.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/parser/ir_parser.h"
#include "paddle/pir/core/utils.h"

using OperatorDialect = paddle::dialect::OperatorDialect;
using AttributeStorage = pir::AttributeStorage;

class TestParserDialect : public pir::Dialect {
 public:
  explicit TestParserDialect(pir::IrContext* context);

  static const char* name() { return "tp"; }

  void PrintAttribute(pir::Attribute attr, std::ostream& os) const;

  pir::Attribute ParseAttribute(pir::IrParser& parser);  // NOLINT

 private:
  void initialize();
};

IR_DECLARE_EXPLICIT_TYPE_ID(TestParserDialect);
IR_DEFINE_EXPLICIT_TYPE_ID(TestParserDialect);

DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(CharAttributeStorage, char);

class CharAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(CharAttribute, CharAttributeStorage);

  char data() const;

  static CharAttribute Parse(pir::IrParser& parser) {  // NOLINT
    std::string char_val = parser.ConsumeToken().val_;
    return CharAttribute::get(parser.ctx, char_val[0]);
  }
};

IR_DECLARE_EXPLICIT_TYPE_ID(CharAttribute);

IR_DEFINE_EXPLICIT_TYPE_ID(CharAttribute);

void TestParserDialect::initialize() { RegisterAttributes<CharAttribute>(); }

char CharAttribute::data() const { return storage()->data(); }

TestParserDialect::TestParserDialect(pir::IrContext* context)
    : pir::Dialect(name(), context, pir::TypeId::get<TestParserDialect>()) {
  initialize();
}

void TestParserDialect::PrintAttribute(pir::Attribute attr,
                                       std::ostream& os) const {
  auto byte_attr = attr.dyn_cast<CharAttribute>();
  os << "(tp.char)" << byte_attr.data();
}

pir::Attribute TestParserDialect::ParseAttribute(
    pir::IrParser& parser) {  // NOLINT
  std::string type_name = parser.ConsumeToken().val_;
  std::string parenthesis_token_val = parser.ConsumeToken().val_;
  IR_ENFORCE(parenthesis_token_val == ")",
             "The token value of expectation is ), not " +
                 parenthesis_token_val + "." + parser.GetErrorLocationInfo());
  return CharAttribute::Parse(parser);
}

TEST(IrParserTest, AddAttribute) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<TestParserDialect>();

  std::string op_str =
      " (%0) = \"builtin.get_parameter\" () "
      "{parameter_name:\"conv2d_0.w_0\",test:(tp.char)a} : () -> "
      "pd_op.tensor<64x3x7x7xf32>";
  std::stringstream ss;
  ss << op_str;
  pir::IrParser* parser = new pir::IrParser(ctx, ss);
  pir::Operation* op = parser->ParseOperation();
  std::stringstream ssp;
  op->Print(ssp);
  delete parser;
  EXPECT_TRUE(ssp.str() == ss.str());
}
