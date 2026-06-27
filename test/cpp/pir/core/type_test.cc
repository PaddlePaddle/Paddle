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

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/type_name.h"
#include "paddle/pir/include/core/type_utils.h"
#include "paddle/pir/include/core/utils.h"
#include "test/cpp/pir/tools/macros_utils.h"

class TypeA {};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TypeA)
IR_DEFINE_EXPLICIT_TYPE_ID(TypeA)

class TypeB {};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TypeB)
IR_DEFINE_EXPLICIT_TYPE_ID(TypeB)

std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

TEST(type_test, type_id) {
  // Test 1: Test construct TypeId by TypeId::get<T>() and overloaded operator==
  // method.
  pir::TypeId a_id = pir::TypeId::get<TypeA>();
  pir::TypeId a_other_id = pir::TypeId::get<TypeA>();
  pir::TypeId b_id = pir::TypeId::get<TypeB>();
  EXPECT_EQ(a_id, a_other_id);
  EXPECT_NE(a_id, b_id);

  // Test 2: Test the hash function of TypeId.
  std::unordered_map<pir::TypeId, pir::TypeId *> type_id_register;
  type_id_register.emplace(a_id, &a_id);
  type_id_register.emplace(b_id, &b_id);
  for (auto kv : type_id_register) {
    EXPECT_EQ(kv.first, *kv.second);
  }
}

// Define a FakeDialect without registering any types.
struct FakeDialect : pir::Dialect {
  explicit FakeDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<FakeDialect>()) {}
  static const char *name() { return "fake"; }
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(FakeDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(FakeDialect)

TEST(type_test, type_base) {
  // Test 1: Test the function of IrContext to register Dialect.
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *fake_dialect = ctx->GetOrRegisterDialect<FakeDialect>();

  // Test 2: Test the get method of AbstractType.
  pir::TypeId a_id = pir::TypeId::get<TypeA>();
  pir::AbstractType abstract_type_a =
      pir::AbstractType::get(a_id, *fake_dialect, {});
  EXPECT_EQ(abstract_type_a.type_id(), a_id);

  // Test 3: Test the constructor of TypeStorage.
  pir::TypeStorage storage_a(&abstract_type_a);
  EXPECT_EQ(storage_a.abstract_type().type_id(), abstract_type_a.type_id());
}

TEST(type_test, built_in_type) {
  // Test the interfaces of class Type: judgment, type_id, abstract_type,
  // classof.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // Test 1: Test the parameterless built-in type of IrContext.
  pir::Type bfp16_1 = pir::BFloat16Type::get(ctx);
  pir::Type bfp16_2 = pir::BFloat16Type::get(ctx);
  EXPECT_EQ(bfp16_1, bfp16_2);
  EXPECT_EQ(bfp16_1.type_id(), bfp16_2.type_id());
  EXPECT_EQ(&bfp16_1.abstract_type(),
            &pir::AbstractType::lookup(bfp16_1.type_id(), ctx));
  EXPECT_EQ(pir::BFloat16Type::classof(bfp16_1), 1);

  pir::Type index_1 = pir::IndexType::get(ctx);
  pir::Type index_2 = pir::IndexType::get(ctx);
  EXPECT_TRUE(index_1.IsIndex());
  EXPECT_EQ(index_1, index_2);
  EXPECT_EQ(index_1.type_id(), index_2.type_id());
  EXPECT_EQ(&index_1.abstract_type(),
            &pir::AbstractType::lookup(index_1.type_id(), ctx));
  EXPECT_EQ(pir::IndexType::classof(index_1), 1);

  pir::Type fp16_1 = pir::Float16Type::get(ctx);
  pir::Type fp16_2 = pir::Float16Type::get(ctx);
  EXPECT_EQ(fp16_1, fp16_2);
  EXPECT_EQ(fp16_1.type_id(), fp16_2.type_id());
  EXPECT_EQ(&fp16_1.abstract_type(),
            &pir::AbstractType::lookup(fp16_1.type_id(), ctx));
  EXPECT_EQ(pir::Float16Type::classof(fp16_1), 1);

  pir::Type fp32_1 = pir::Float32Type::get(ctx);
  pir::Type fp32_2 = pir::Float32Type::get(ctx);
  EXPECT_EQ(fp32_1, fp32_2);
  EXPECT_EQ(fp32_1.type_id(), fp32_2.type_id());
  EXPECT_EQ(&fp32_1.abstract_type(),
            &pir::AbstractType::lookup(fp32_1.type_id(), ctx));
  EXPECT_EQ(pir::Float32Type::classof(fp32_1), 1);

  pir::Type fp64_1 = pir::Float64Type::get(ctx);
  pir::Type fp64_2 = pir::Float64Type::get(ctx);
  EXPECT_EQ(fp64_1, fp64_2);
  EXPECT_EQ(fp64_1.type_id(), fp64_2.type_id());
  EXPECT_EQ(&fp64_1.abstract_type(),
            &pir::AbstractType::lookup(fp64_1.type_id(), ctx));
  EXPECT_EQ(pir::Float64Type::classof(fp64_1), 1);

  pir::Type int16_1 = pir::Int16Type::get(ctx);
  pir::Type int16_2 = pir::Int16Type::get(ctx);
  EXPECT_EQ(int16_1, int16_2);
  EXPECT_EQ(int16_1.type_id(), int16_2.type_id());
  EXPECT_EQ(&int16_1.abstract_type(),
            &pir::AbstractType::lookup(int16_1.type_id(), ctx));
  EXPECT_EQ(pir::Int16Type::classof(int16_1), 1);

  pir::Type int32_1 = pir::Int32Type::get(ctx);
  pir::Type int32_2 = pir::Int32Type::get(ctx);
  EXPECT_EQ(int32_1, int32_2);
  EXPECT_EQ(int32_1.type_id(), int32_2.type_id());
  EXPECT_EQ(&int32_1.abstract_type(),
            &pir::AbstractType::lookup(int32_1.type_id(), ctx));
  EXPECT_EQ(pir::Int32Type::classof(int32_1), 1);

  pir::Type int64_1 = pir::Int64Type::get(ctx);
  pir::Type int64_2 = pir::Int64Type::get(ctx);
  EXPECT_EQ(int64_1, int64_2);
  EXPECT_EQ(int64_1.type_id(), int64_2.type_id());
  EXPECT_EQ(&int64_1.abstract_type(),
            &pir::AbstractType::lookup(int64_1.type_id(), ctx));
  EXPECT_EQ(pir::Int64Type::classof(int64_1), 1);

  // Test 2: Test isa and dyn_cast.
  EXPECT_EQ(fp16_1.isa<pir::Float16Type>(), true);
  EXPECT_EQ(fp16_1.isa<pir::Float32Type>(), false);
  EXPECT_EQ(fp16_1.isa<pir::Type>(), true);

  // Test 3: Test VectorType
  std::vector<pir::Type> vec_type = {int32_1, int64_1};
  pir::Type vector_type = pir::VectorType::get(ctx, vec_type);
  EXPECT_EQ(vector_type.isa<pir::VectorType>(), true);
  EXPECT_EQ(vector_type.dyn_cast<pir::VectorType>().size() == 2, true);
  EXPECT_EQ(vector_type.dyn_cast<pir::VectorType>()[0].isa<pir::Int32Type>(),
            true);
  EXPECT_EQ(vector_type.dyn_cast<pir::VectorType>()[1].isa<pir::Int64Type>(),
            true);
}

// Customize a parameterized TypeStorage IntegerTypeStorage.
struct IntegerTypeStorage : public pir::TypeStorage {
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

  static IntegerTypeStorage *Construct(const ParamKey &key) {
    return new IntegerTypeStorage(key.first, key.second);
  }

  ParamKey GetAsKey() const { return ParamKey(width_, signedness_); }

  unsigned width_ : 30;
  unsigned signedness_ : 2;
};

// Customize a parameterized type: IntegerType, storage type is
// IntegerTypeStorage.
class IntegerType
    : public pir::Type::TypeBase<IntegerType, pir::Type, IntegerTypeStorage> {
 public:
  using Base::Base;
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(IntegerType)
IR_DEFINE_EXPLICIT_TYPE_ID(IntegerType)

// Customize a Dialect IntegerDialect, registration type of IntegerType.
struct IntegerDialect : pir::Dialect {
  explicit IntegerDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<IntegerDialect>()) {
    RegisterType<IntegerType>();
  }
  static const char *name() { return "integer"; }
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(IntegerDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(IntegerDialect)

TEST(type_test, custom_type_dialect) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  // Test 1: Test the function of IrContext to register Dialect.
  ctx->GetOrRegisterDialect<IntegerDialect>();

  pir::Type int1_1 = IntegerType::get(ctx, 1, 0);
  pir::Type int1_2 = IntegerType::get(ctx, 1, 0);
  EXPECT_EQ(int1_1, int1_2);

  pir::Type int8 = IntegerType::get(ctx, 8, 0);
  EXPECT_NE(int8, int1_2);

  //  Test 2: Test Dialect interfaces
  EXPECT_EQ(ctx, int8.ir_context());

  EXPECT_EQ(int8.dialect().id(), pir::TypeId::get<IntegerDialect>());

  std::vector<pir::Dialect *> dialect_list = ctx->GetRegisteredDialects();
  EXPECT_EQ(dialect_list.size() == 5, 1);  // integer, builtin, fake

  pir::Dialect *dialect_builtin1 = ctx->GetRegisteredDialect("builtin");
  pir::Dialect *dialect_builtin2 =
      ctx->GetRegisteredDialect<pir::BuiltinDialect>();
  EXPECT_EQ(dialect_builtin1, dialect_builtin2);

  pir::Dialect *dialect_integer1 = ctx->GetRegisteredDialect("integer");
  pir::Dialect *dialect_integer2 = ctx->GetRegisteredDialect<IntegerDialect>();
  EXPECT_EQ(dialect_integer1, dialect_integer2);
}

TEST(type_test, sparse_coo) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  common::DDim dims = {4, 4};
  common::DDim non_zero_dims = {4, 1};
  common::DataLayout data_layout = common::DataLayout::NCHW;
  pir::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType none_zero_indices = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);
  pir::DenseTensorType none_zero_elements = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);
  bool coalesced = false;
  paddle::dialect::SparseCooTensorTypeStorage storage1(fp32_dtype,
                                                       dims,
                                                       non_zero_dims,
                                                       data_layout,
                                                       none_zero_indices,
                                                       none_zero_elements,
                                                       coalesced);
  auto storage2 = std::make_tuple(fp32_dtype,
                                  dims,
                                  non_zero_dims,
                                  data_layout,
                                  none_zero_indices,
                                  none_zero_elements,
                                  coalesced);
  pir::Type pir_type =
      paddle::dialect::SparseCooTensorType::get(ctx,
                                                fp32_dtype,
                                                dims,
                                                non_zero_dims,
                                                data_layout,
                                                none_zero_indices,
                                                none_zero_elements,
                                                coalesced);
  EXPECT_TRUE(storage1 == storage2);
  EXPECT_EQ(pir_type.isa<paddle::dialect::SparseCooTensorType>(), true);
  paddle::dialect::SparseCooTensorType sparse_coo_tensor_type =
      pir_type.dyn_cast<paddle::dialect::SparseCooTensorType>();
  EXPECT_EQ(sparse_coo_tensor_type.dims(), dims);
  EXPECT_EQ(sparse_coo_tensor_type.non_zero_dims(), non_zero_dims);
  EXPECT_EQ(sparse_coo_tensor_type.data_layout(), data_layout);
  EXPECT_EQ(sparse_coo_tensor_type.non_zero_indices(), none_zero_indices);
  EXPECT_EQ(sparse_coo_tensor_type.non_zero_elements(), none_zero_elements);
  EXPECT_EQ(sparse_coo_tensor_type.coalesced(), coalesced);
}

TEST(type_test, pd_op_dialect) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  paddle::dialect::SelectedRowsType select_rows_dtype =
      paddle::dialect::SelectedRowsType::get(
          ctx, fp32_dtype, dims, data_layout, lod, offset);
  EXPECT_EQ(select_rows_dtype.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(select_rows_dtype.dims(), dims);
  EXPECT_EQ(select_rows_dtype.data_layout(), data_layout);
  EXPECT_EQ(select_rows_dtype.lod(), lod);
  EXPECT_EQ(select_rows_dtype.offset(), offset);
}

TEST(type_test, sparse_csr) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  common::DDim dims = {4, 4};
  common::DataLayout data_layout = common::DataLayout::NCHW;
  pir::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType non_zero_crows = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);
  pir::DenseTensorType non_zero_cols = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);
  pir::DenseTensorType non_zero_elements = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);
  paddle::dialect::SparseCsrTensorTypeStorage storage1(fp32_dtype,
                                                       dims,
                                                       data_layout,
                                                       non_zero_crows,
                                                       non_zero_cols,
                                                       non_zero_elements);
  auto storage2 = std::make_tuple(fp32_dtype,
                                  dims,
                                  data_layout,
                                  non_zero_crows,
                                  non_zero_cols,
                                  non_zero_elements);
  pir::Type pir_type =
      paddle::dialect::SparseCsrTensorType::get(ctx,
                                                fp32_dtype,
                                                dims,
                                                data_layout,
                                                non_zero_crows,
                                                non_zero_cols,
                                                non_zero_elements);
  EXPECT_TRUE(storage1 == storage2);
  EXPECT_EQ(pir_type.isa<paddle::dialect::SparseCsrTensorType>(), true);
  paddle::dialect::SparseCsrTensorType sparse_csr_tensor_type =
      pir_type.dyn_cast<paddle::dialect::SparseCsrTensorType>();
  EXPECT_EQ(sparse_csr_tensor_type.dims(), dims);
  EXPECT_EQ(sparse_csr_tensor_type.data_layout(), data_layout);
  EXPECT_EQ(sparse_csr_tensor_type.non_zero_crows(), non_zero_crows);
  EXPECT_EQ(sparse_csr_tensor_type.non_zero_cols(), non_zero_cols);
  EXPECT_EQ(sparse_csr_tensor_type.non_zero_elements(), non_zero_elements);
}

TEST(type_test, type_util) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims1 = {2, 2};
  phi::DDim dims2 = {2, 2, 3};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  paddle::dialect::SelectedRowsType select_rows_dtype1 =
      paddle::dialect::SelectedRowsType::get(
          ctx, fp32_dtype, dims1, data_layout, lod, offset);

  paddle::dialect::SelectedRowsType select_rows_dtype2 =
      paddle::dialect::SelectedRowsType::get(
          ctx, fp32_dtype, dims2, data_layout, lod, offset);

  std::vector<pir::Type> types1 = {
      select_rows_dtype1, select_rows_dtype1, select_rows_dtype1};
  std::vector<pir::Type> types2 = {
      select_rows_dtype1, select_rows_dtype1, select_rows_dtype1};
  std::vector<pir::Type> types3 = {
      select_rows_dtype2, select_rows_dtype2, select_rows_dtype2};

  EXPECT_TRUE(pir::VerifyCompatibleShapes(types1, types2));
  EXPECT_FALSE(pir::VerifyCompatibleShapes(types1, types3));
}

namespace TestNamespace {
class TestClass {};
}  // namespace TestNamespace

TEST(type_test, get_type_name) {
  auto name = pir::get_type_name<TestNamespace::TestClass>();
  EXPECT_EQ(name, "TestNamespace::TestClass");
}
