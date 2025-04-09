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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_attribute_storage.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/utils.h"
#include "test/cpp/pir/tools/macros_utils.h"

using OperatorDialect = paddle::dialect::OperatorDialect;
using AttributeStorage = pir::AttributeStorage;

class TestParserDialect : public pir::Dialect {
 public:
  explicit TestParserDialect(pir::IrContext* context);

  static const char* name() { return "tp"; }

  void PrintAttribute(pir::Attribute attr, std::ostream& os) const;  // NOLINT

 private:
  void initialize();
};

IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TestParserDialect);
IR_DEFINE_EXPLICIT_TYPE_ID(TestParserDialect);

DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(CharAttributeStorage, char);

class CharAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(CharAttribute, CharAttributeStorage);

  char data() const;
};

IR_DECLARE_EXPLICIT_TEST_TYPE_ID(CharAttribute);

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
