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
#include <unordered_map>

#include "paddle/ir/attribute.h"
#include "paddle/ir/attribute_base.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"

TEST(attribute_test, attribute_base) {
  // Define two empty classes, just for testing.
  class AttributeA {};

  // Define a FakeDialect without registering any types.
  struct FakeDialect : ir::Dialect {
    explicit FakeDialect(ir::IrContext *context)
        : ir::Dialect(name(), context, ir::TypeId::get<FakeDialect>()) {}
    static const char *name() { return "fake"; }
  };

  // Test 1: Test the function of IrContext to register Dialect.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *fake_dialect = ctx->GetOrRegisterDialect<FakeDialect>();

  // Test 2: Test the get method of AbstractType.
  ir::TypeId a_id = ir::TypeId::get<AttributeA>();
  ir::AbstractAttribute abstract_attribute_a =
      ir::AbstractAttribute::get(a_id, *fake_dialect);
  EXPECT_EQ(abstract_attribute_a.type_id(), a_id);

  // Test 3: Test the constructor of TypeStorage.
  ir::AttributeStorage storage_a(&abstract_attribute_a);
  EXPECT_EQ(storage_a.abstract_attribute().type_id(),
            abstract_attribute_a.type_id());
}

TEST(attribute_test, built_in_attribute) {
  // Test the interfaces of class Attribute: judgment, type_id,
  // abstract_attribute, classof.
  ir::IrContext *ctx = ir::IrContext::Instance();

  // Test 1: Test the parameteric built-in attribute of IrContext.
  ir::Attribute string_attr_1 = ir::StringAttribute::get(ctx, "string_a", 8);
  ir::Attribute string_attr_2 = ir::StringAttribute::get(ctx, "string_a", 8);
  ir::Attribute string_attr_3 = ir::StringAttribute::get(ctx, "string_b", 8);

  EXPECT_EQ(string_attr_1, string_attr_2);
  EXPECT_NE(string_attr_1, string_attr_3);
  EXPECT_EQ(string_attr_1.type_id(), string_attr_2.type_id());
  EXPECT_EQ(ir::StringAttribute::classof(string_attr_1), 1);

  // Test 2: Test isa and dyn_cast.
  EXPECT_EQ(string_attr_1.isa<ir::StringAttribute>(), true);

  ir::StringAttribute string_attr_cast_1 =
      string_attr_1.dyn_cast<ir::StringAttribute>();
  EXPECT_EQ(string_attr_cast_1.isa<ir::StringAttribute>(), true);
  EXPECT_EQ(string_attr_cast_1.size() == 8, 1);
}
