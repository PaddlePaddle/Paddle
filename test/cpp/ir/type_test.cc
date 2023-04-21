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

#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/type.h"
#include "paddle/ir/type_base.h"
#include "paddle/ir/utils.h"

TEST(type_test, type_id) {
  // Define two empty classes, just for testing.
  class TypeA {};
  class TypeB {};

  // Test 1: Test construct TypeId by TypeId::get<T>() and overloaded operator==
  // method.
  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::TypeId a_other_id = ir::TypeId::get<TypeA>();
  ir::TypeId b_id = ir::TypeId::get<TypeB>();
  EXPECT_EQ(a_id, a_other_id);
  EXPECT_NE(a_id, b_id);

  // Test 2: Test the hash function of TypeId.
  std::unordered_map<ir::TypeId, ir::TypeId *> type_id_register;
  type_id_register.emplace(a_id, &a_id);
  type_id_register.emplace(b_id, &b_id);
  for (auto kv : type_id_register) {
    EXPECT_EQ(kv.first, *kv.second);
  }
}

TEST(type_test, type_base) {
  // Define two empty classes, just for testing.
  class TypeA {};

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
  ir::TypeId a_id = ir::TypeId::get<TypeA>();
  ir::AbstractType abstract_type_a = ir::AbstractType::get(a_id, *fake_dialect);
  EXPECT_EQ(abstract_type_a.type_id(), a_id);

  // Test 3: Test the constructor of TypeStorage.
  ir::TypeStorage storage_a(&abstract_type_a);
  EXPECT_EQ(storage_a.abstract_type().type_id(), abstract_type_a.type_id());
}

TEST(type_test, built_in_type) {
  // Test the interfaces of class Type: judgment, type_id, abstract_type,
  // classof.
  ir::IrContext *ctx = ir::IrContext::Instance();

  // Test 1: Test the parameterless built-in type of IrContext.
  ir::Type fp16_1 = ir::Float16Type::get(ctx);
  ir::Type fp16_2 = ir::Float16Type::get(ctx);
  EXPECT_EQ(fp16_1, fp16_2);
  EXPECT_EQ(fp16_1.type_id(), fp16_2.type_id());
  EXPECT_EQ(&fp16_1.abstract_type(),
            &ir::AbstractType::lookup(fp16_1.type_id(), ctx));
  EXPECT_EQ(ir::Float16Type::classof(fp16_1), 1);

  ir::Type fp32_1 = ir::Float32Type::get(ctx);
  ir::Type fp32_2 = ir::Float32Type::get(ctx);
  EXPECT_EQ(fp32_1, fp32_2);
  EXPECT_EQ(fp32_1.type_id(), fp32_2.type_id());
  EXPECT_EQ(&fp32_1.abstract_type(),
            &ir::AbstractType::lookup(fp32_1.type_id(), ctx));
  EXPECT_EQ(ir::Float32Type::classof(fp32_1), 1);

  ir::Type fp64_1 = ir::Float64Type::get(ctx);
  ir::Type fp64_2 = ir::Float64Type::get(ctx);
  EXPECT_EQ(fp64_1, fp64_2);
  EXPECT_EQ(fp64_1.type_id(), fp64_2.type_id());
  EXPECT_EQ(&fp64_1.abstract_type(),
            &ir::AbstractType::lookup(fp64_1.type_id(), ctx));
  EXPECT_EQ(ir::Float64Type::classof(fp64_1), 1);

  ir::Type int16_1 = ir::Int16Type::get(ctx);
  ir::Type int16_2 = ir::Int16Type::get(ctx);
  EXPECT_EQ(int16_1, int16_2);
  EXPECT_EQ(int16_1.type_id(), int16_2.type_id());
  EXPECT_EQ(&int16_1.abstract_type(),
            &ir::AbstractType::lookup(int16_1.type_id(), ctx));
  EXPECT_EQ(ir::Int16Type::classof(int16_1), 1);

  ir::Type int32_1 = ir::Int32Type::get(ctx);
  ir::Type int32_2 = ir::Int32Type::get(ctx);
  EXPECT_EQ(int32_1, int32_2);
  EXPECT_EQ(int32_1.type_id(), int32_2.type_id());
  EXPECT_EQ(&int32_1.abstract_type(),
            &ir::AbstractType::lookup(int32_1.type_id(), ctx));
  EXPECT_EQ(ir::Int32Type::classof(int32_1), 1);

  ir::Type int64_1 = ir::Int64Type::get(ctx);
  ir::Type int64_2 = ir::Int64Type::get(ctx);
  EXPECT_EQ(int64_1, int64_2);
  EXPECT_EQ(int64_1.type_id(), int64_2.type_id());
  EXPECT_EQ(&int64_1.abstract_type(),
            &ir::AbstractType::lookup(int64_1.type_id(), ctx));
  EXPECT_EQ(ir::Int64Type::classof(int64_1), 1);

  // Test 2: Test the parameteric built-in type of IrContext.
  ir::DenseTensorTypeStorage::Dim dims = {1, 2, 3};
  ir::DenseTensorTypeStorage::DataLayout data_layout =
      ir::DenseTensorTypeStorage::DataLayout::NCHW;
  ir::DenseTensorTypeStorage::LoD lod = {{1, 2, 3}, {4, 5, 6}};
  size_t offset = 0;

  ir::Type dense_tensor_1 =
      ir::DenseTensorType::get(ctx, fp32_1, dims, data_layout, lod, offset);
  ir::Type dense_tensor_2 =
      ir::DenseTensorType::get(ctx, fp32_2, dims, data_layout, lod, offset);
  ir::Type dense_tensor_3 =
      ir::DenseTensorType::get(ctx, fp32_1, dims, data_layout, lod, 2);

  EXPECT_EQ(dense_tensor_1, dense_tensor_2);
  EXPECT_NE(dense_tensor_1, dense_tensor_3);
  EXPECT_EQ(dense_tensor_1.type_id(), dense_tensor_2.type_id());
  EXPECT_EQ(ir::DenseTensorType::classof(dense_tensor_1), 1);

  ir::DenseTensorType dense_tensor_4 =
      ir::DenseTensorType::get(ctx, fp32_1, dims, data_layout, lod, 2);
  EXPECT_EQ(dense_tensor_4.offset() == 2, 1);
  EXPECT_EQ(dense_tensor_4.dtype().isa<ir::Float32Type>(), true);
  EXPECT_EQ(dense_tensor_4.data_layout(), data_layout);

  // Test 3: Test isa and dyn_cast.
  EXPECT_EQ(fp16_1.isa<ir::Float16Type>(), true);
  EXPECT_EQ(fp16_1.isa<ir::Float32Type>(), false);
  EXPECT_EQ(fp16_1.isa<ir::DenseTensorType>(), false);
  EXPECT_EQ(fp16_1.isa<ir::Type>(), true);
  EXPECT_EQ(dense_tensor_1.isa<ir::DenseTensorType>(), true);

  ir::DenseTensorType dense_tensor_cast_1 =
      dense_tensor_1.dyn_cast<ir::DenseTensorType>();
  EXPECT_EQ(dense_tensor_cast_1.isa<ir::DenseTensorType>(), true);
  EXPECT_EQ(dense_tensor_cast_1.offset() == 0, 1);
  const ir::DenseTensorType dense_tensor_cast_2 =
      ir::dyn_cast<ir::DenseTensorType>(dense_tensor_1);
  EXPECT_EQ(dense_tensor_cast_2.isa<ir::DenseTensorType>(), true);
  EXPECT_EQ(dense_tensor_cast_2.offset() == 0, 1);
}

// Customize a parameterized TypeStorage IntegerTypeStorage.
struct IntegerTypeStorage : public ir::TypeStorage {
  IntegerTypeStorage(unsigned width, unsigned signedness)
      : width_(width), signedness_(signedness) {}
  using ParamKey = std::pair<unsigned, unsigned>;

  static std::size_t HashValue(const ParamKey &key) {
    return ir::hash_combine(std::hash<unsigned>()(std::get<0>(key)),
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
};

// Customize a parameterized type: IntegerType, storage type is
// IntegerTypeStorage.
class IntegerType : public ir::Type {
 public:
  using Type::Type;
  DECLARE_TYPE_UTILITY_FUNCTOR(IntegerType, IntegerTypeStorage);
};

// Customize a Dialect IntegerDialect, registration type of IntegerType.
struct IntegerDialect : ir::Dialect {
  explicit IntegerDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<IntegerDialect>()) {
    RegisterType<IntegerType>();
  }
  static const char *name() { return "integer"; }
};

TEST(type_test, custom_type_dialect) {
  ir::IrContext *ctx = ir::IrContext::Instance();

  // Test 1: Test the function of IrContext to register Dialect.
  ctx->GetOrRegisterDialect<IntegerDialect>();

  ir::Type int1_1 = IntegerType::get(ctx, 1, 0);
  ir::Type int1_2 = IntegerType::get(ctx, 1, 0);
  EXPECT_EQ(int1_1, int1_2);

  ir::Type int8 = IntegerType::get(ctx, 8, 0);
  EXPECT_NE(int8, int1_2);

  //  Test 2: Test Dialect interfaces
  EXPECT_EQ(ctx, int8.ir_context());

  EXPECT_EQ(int8.dialect().id(), ir::TypeId::get<IntegerDialect>());

  std::vector<ir::Dialect *> dialect_list = ctx->GetRegisteredDialects();
  EXPECT_EQ(dialect_list.size() == 3, 1);  // integer, builtin, fake

  ir::Dialect *dialect_builtin1 = ctx->GetRegisteredDialect("builtin");
  ir::Dialect *dialect_builtin2 =
      ctx->GetRegisteredDialect<ir::BuiltinDialect>();
  EXPECT_EQ(dialect_builtin1, dialect_builtin2);

  ir::Dialect *dialect_integer1 = ctx->GetRegisteredDialect("integer");
  ir::Dialect *dialect_integer2 = ctx->GetRegisteredDialect<IntegerDialect>();
  EXPECT_EQ(dialect_integer1, dialect_integer2);
}
