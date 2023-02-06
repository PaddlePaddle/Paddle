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

#include "paddle/fluid/framework/ir/type/type_utils.h"
#include <gtest/gtest.h>
#include <unordered_map>

TEST(type_utils, type_id) {
  class TypeA {};
  class TypeB {};
  paddle::framework::ir::TypeID a_id =
      paddle::framework::ir::TypeID::Get<TypeA>();
  paddle::framework::ir::TypeID a_other_id =
      paddle::framework::ir::TypeID::Get<TypeA>();
  paddle::framework::ir::TypeID b_id =
      paddle::framework::ir::TypeID::Get<TypeB>();
  const void* type_a_storage = a_id.GetStorage();
  const void* type_b_storage = b_id.GetStorage();
  EXPECT_EQ(a_id, a_other_id);
  EXPECT_NE(a_id, b_id);

  std::unordered_map<paddle::framework::ir::TypeID, const void*>
      type_id_register;
  type_id_register.emplace(a_id, type_a_storage);
  type_id_register.emplace(b_id, type_b_storage);
  for (auto kv : type_id_register) {
    EXPECT_EQ(kv.first.GetStorage(), kv.second);
  }
}
