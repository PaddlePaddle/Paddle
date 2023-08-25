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
#include "paddle/ir/core/utils.h"

using PaddleDialect = paddle::dialect::PaddleDialect;
using AttributeStorage = ir::AttributeStorage;

class TestParserDialect : public ir::Dialect {
 public:
  explicit TestParserDialect(ir::IrContext* context);

  static const char* name() { return "tp"; }

  void PrintAttribute(ir::Attribute attr, std::ostream& os) const;

  ir::Attribute ParseAttribute(ir::IrParser& parser);  // NOLINT

 private:
  void initialize();
};

IR_DECLARE_EXPLICIT_TYPE_ID(TestParserDialect);
IR_DEFINE_EXPLICIT_TYPE_ID(TestParserDialect);

DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(CharAttributeStorage, char);

class CharAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(CharAttribute, CharAttributeStorage);

  char data() const;

  static CharAttribute Parse(ir::IrParser& parser) {  // NOLINT
    std::string char_val = parser.GetToken().val_;
    return CharAttribute::get(parser.ctx, char_val[0]);
  }
};

IR_DECLARE_EXPLICIT_TYPE_ID(CharAttribute);

IR_DEFINE_EXPLICIT_TYPE_ID(CharAttribute);

void TestParserDialect::initialize() { RegisterAttributes<CharAttribute>(); }

char CharAttribute::data() const { return storage()->data(); }

TestParserDialect::TestParserDialect(ir::IrContext* context)
    : ir::Dialect(name(), context, ir::TypeId::get<TestParserDialect>()) {
  initialize();
}

void TestParserDialect::PrintAttribute(ir::Attribute attr,
                                       std::ostream& os) const {
  auto byte_attr = attr.dyn_cast<CharAttribute>();
  os << "(tp.char)" << byte_attr.data();
}

ir::Attribute TestParserDialect::ParseAttribute(
    ir::IrParser& parser) {  // NOLINT
  std::string type_name = parser.GetToken().val_;
  std::string parenthesis_token_val = parser.GetToken().val_;
  IR_ENFORCE(parenthesis_token_val == ")",
             "The token value of expectation is ), not " +
                 parenthesis_token_val + "." + parser.GetErrorLocationInfo());
  return CharAttribute::Parse(parser);
}

TEST(IrParserTest, AddAttribute) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<TestParserDialect>();

  string op_str =
      " (%0) = \"builtin.get_parameter\" () "
      "{parameter_name:(String)conv2d_0.w_0,test:(tp.char)a} : () -> "
      "pd.tensor<64x3x7x7xf32>";
  std::stringstream ss;
  ss << op_str;
  ir::IrParser* parser = new ir::IrParser(ctx, ss);
  ir::Operation* op = parser->ParseOperation();
  std::stringstream ssp;
  op->Print(ssp);
  EXPECT_TRUE(ssp.str() == ss.str());
}
