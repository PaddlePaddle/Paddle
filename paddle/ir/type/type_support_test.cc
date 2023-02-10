/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/ir/type/type_support.h"
#include <gtest/gtest.h>
#include <unordered_map>

TEST(type_support, type_id) {
  class TypeA {};
  class TypeB {};

  // (1) Test construct TypeId by TypeId::Get()
  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::TypeId a_other_id = ir::TypeId::get<TypeA>();
  ir::TypeId b_id = ir::TypeId::get<TypeB>();
  EXPECT_EQ(a_id, a_other_id);
  EXPECT_NE(a_id, b_id);

  // (2) Test TypeId hash
  std::unordered_map<ir::TypeId, ir::TypeId*> type_id_register;
  type_id_register.emplace(a_id, &a_id);
  type_id_register.emplace(b_id, &b_id);
  for (auto kv : type_id_register) {
    EXPECT_EQ(kv.first, *kv.second);
  }
}

TEST(type_support, abstract_type) {
  class TypeA {};

  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::AbstractType abstract_type_a = ir::AbstractType::get(a_id);

  EXPECT_EQ(abstract_type_a.type_id(), a_id);
}

TEST(type_support, type_storage) {
  class TypeA {};

  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::AbstractType abstract_type_a = ir::AbstractType::get(a_id);

  ir::TypeStorage storage_a(&abstract_type_a);

  EXPECT_EQ(storage_a.abstract_type().type_id(), abstract_type_a.type_id());
}
