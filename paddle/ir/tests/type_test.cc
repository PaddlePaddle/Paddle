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

#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/type_base.h"

TEST(type_test, type_id) {
  class TypeA {};
  class TypeB {};

  // (1) Test construct TypeId by TypeId::Get()
  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::TypeId a_other_id = ir::TypeId::get<TypeA>();
  ir::TypeId b_id = ir::TypeId::get<TypeB>();
  EXPECT_EQ(a_id, a_other_id);
  EXPECT_NE(a_id, b_id);

  // (2) Test TypeId hash
  std::unordered_map<ir::TypeId, ir::TypeId *> type_id_register;
  type_id_register.emplace(a_id, &a_id);
  type_id_register.emplace(b_id, &b_id);
  for (auto kv : type_id_register) {
    EXPECT_EQ(kv.first, *kv.second);
  }
}

TEST(type_test, abstract_type) {
  class TypeA {};

  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::AbstractType abstract_type_a = ir::AbstractType::get(a_id);

  EXPECT_EQ(abstract_type_a.type_id(), a_id);
}

TEST(type_test, type_storage) {
  class TypeA {};

  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::AbstractType abstract_type_a = ir::AbstractType::get(a_id);

  ir::TypeStorage storage_a(&abstract_type_a);

  EXPECT_EQ(storage_a.abstract_type().type_id(), abstract_type_a.type_id());
}

TEST(type_test, built_in_type) {
  // Test creation of built-in parameterless type.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Type fp32_1 = ir::Float32Type::get(ctx);

  // Test interfaces of class Type
  ir::Type fp32_2 = ir::Float32Type::get(ctx);
  EXPECT_EQ(fp32_1 == fp32_2, 1);
  EXPECT_EQ(fp32_1 != fp32_2, 0);
  EXPECT_EQ(fp32_1.type_id() == fp32_2.type_id(), 1);
  EXPECT_EQ(&fp32_1.abstract_type() ==
                &ir::AbstractType::lookup(fp32_1.type_id(), ctx),
            1);
  EXPECT_EQ(ir::Float32Type::classof(fp32_1), 1);

  ir::Type int32_1 = ir::Int32Type::get(ctx);
  ir::Type int32_2 = ir::Int32Type::get(ctx);
  EXPECT_EQ(int32_1 == int32_2, 1);
  EXPECT_EQ(int32_1.type_id() == int32_2.type_id(), 1);
  EXPECT_EQ(&int32_1.abstract_type() ==
                &ir::AbstractType::lookup(int32_1.type_id(), ctx),
            1);
  EXPECT_EQ(ir::Int32Type::classof(int32_1), 1);
}

struct IntegerTypeStorage : public ir::TypeStorage {
  IntegerTypeStorage(unsigned width, unsigned signedness)
      : width_(width), signedness_(signedness) {}
  using ParamKey = std::pair<unsigned, unsigned>;

  static std::size_t HashValue(const ParamKey &key) {
    return hash_combine(std::hash<unsigned>()(std::get<0>(key)),
                        std::hash<unsigned>()(std::get<1>(key)));
  }

  bool operator==(const ParamKey &key) const {
    return ParamKey(width_, signedness_) == key;
  }

  static IntegerTypeStorage *Construct(ParamKey key) {
    return new IntegerTypeStorage(key.first, key.second);
  }

  ParamKey GetAsKey() const { return ParamKey(width_, signedness_); }

  unsigned width_ : 30;
  unsigned signedness_ : 2;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

class IntegerType : public ir::Type {
 public:
  using Type::Type;
  DECLARE_TYPE_UTILITY_FUNCTOR(IntegerType, IntegerTypeStorage);
};

TEST(type_test, parameteric_type) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  REGISTER_TYPE_2_IRCONTEXT(IntegerType, ctx);
  ir::Type int1_1 = IntegerType::get(ctx, 1, 0);
  ir::Type int1_2 = IntegerType::get(ctx, 1, 0);
  EXPECT_EQ(int1_1 == int1_2, 1);

  ir::Type int8 = IntegerType::get(ctx, 8, 0);
  EXPECT_EQ(int8 == int1_2, 0);
}
