// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/typeid.h>

#include <array>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

namespace {

struct LifecycleTrackedType {
  LifecycleTrackedType() = default;
  explicit LifecycleTrackedType(int new_value) : value(new_value) {}

  int value{7};
};

struct NonDefaultConstructibleType {
  explicit NonDefaultConstructibleType(int new_value) : value(new_value) {}

  int value;
};

struct NonCopyAssignableType {
  NonCopyAssignableType() = default;
  explicit NonCopyAssignableType(int new_value) : value(new_value) {}
  NonCopyAssignableType(const NonCopyAssignableType&) = default;
  NonCopyAssignableType& operator=(const NonCopyAssignableType&) = delete;

  int value{5};
};

caffe2::TypeIdentifier GetIntTypeIdentifierFromHelper() {
  return caffe2::TypeIdentifier::Get<int>();
}

caffe2::TypeMeta MakeStdStringTypeMetaFromHelper() {
  return caffe2::TypeMeta::Make<std::string>();
}

}  // namespace

namespace caffe2 {

CAFFE_KNOWN_TYPE_NOEXPORT(::LifecycleTrackedType)
CAFFE_KNOWN_TYPE_NOEXPORT(::NonDefaultConstructibleType)
CAFFE_KNOWN_TYPE_NOEXPORT(::NonCopyAssignableType)

}  // namespace caffe2

TEST(TypeIdentifierCompatTest, SameTypeHasStableId) {
  const auto id1 = caffe2::TypeIdentifier::Get<int>();
  const auto id2 = caffe2::TypeIdentifier::Get<int>();
  const auto id3 = GetIntTypeIdentifierFromHelper();

  EXPECT_EQ(id1, id2);
  EXPECT_EQ(id1, id3);
}

TEST(TypeIdentifierCompatTest, DifferentTypeHasDifferentId) {
  const auto int_id = caffe2::TypeIdentifier::Get<int>();
  const auto float_id = caffe2::TypeIdentifier::Get<float>();

  EXPECT_NE(int_id, float_id);
}

TEST(TypeIdentifierCompatTest, OrderingHashAndStreamWork) {
  const auto int_id = caffe2::TypeIdentifier::Get<int>();
  const auto string_id = caffe2::TypeIdentifier::Get<std::string>();
  const auto uninitialized = caffe2::TypeIdentifier::uninitialized();

  EXPECT_EQ(uninitialized.underlyingId(), 0U);
  EXPECT_NE(std::hash<caffe2::TypeIdentifier>{}(int_id),
            std::hash<caffe2::TypeIdentifier>{}(uninitialized));
  EXPECT_TRUE(uninitialized < int_id || int_id < uninitialized);

  std::ostringstream stream;
  stream << int_id;
  EXPECT_FALSE(stream.str().empty());
  EXPECT_NE(stream.str(), "0");
  EXPECT_NE(int_id, string_id);
}

TEST(TypeMetaCompatTest, ScalarTypeRoundTrip) {
  const std::array<c10::ScalarType, 6> dtypes = {
      c10::ScalarType::Bool,
      c10::ScalarType::Half,
      c10::ScalarType::Float,
      c10::ScalarType::Double,
      c10::ScalarType::Int,
      c10::ScalarType::Long,
  };

  for (const auto dtype : dtypes) {
    const auto type_meta = caffe2::TypeMeta::fromScalarType(dtype);
    EXPECT_TRUE(type_meta.isScalarType(dtype));
    EXPECT_EQ(type_meta.toScalarType(), dtype);
  }
}

TEST(TypeMetaCompatTest, ScalarTypeHelperConversionsAndComparisons) {
  const auto type_meta = c10::scalarTypeToTypeMeta(c10::kDouble);

  EXPECT_EQ(c10::typeMetaToScalarType(type_meta), c10::kDouble);
  EXPECT_EQ(c10::optTypeMetaToScalarType(std::make_optional(type_meta)),
            std::make_optional(c10::kDouble));
  EXPECT_EQ(c10::optTypeMetaToScalarType(std::optional<caffe2::TypeMeta>{}),
            std::nullopt);
  EXPECT_TRUE(type_meta == c10::kDouble);
  EXPECT_TRUE(c10::kDouble == type_meta);
  EXPECT_TRUE(type_meta != c10::kFloat);
  EXPECT_TRUE(c10::kFloat != type_meta);
}

TEST(TypeMetaCompatTest, BuiltinKnownTypeIsStableAcrossTranslationUnits) {
  const auto local_meta = caffe2::TypeMeta::Make<std::string>();
  const auto helper_meta = MakeStdStringTypeMetaFromHelper();

  EXPECT_EQ(local_meta, helper_meta);
  EXPECT_EQ(local_meta.id(), helper_meta.id());
}

TEST(TypeMetaCompatTest, NonScalarTypeToScalarTypeThrows) {
  const auto non_scalar_meta = caffe2::TypeMeta::Make<std::string>();

  EXPECT_FALSE(non_scalar_meta.isScalarType());
  EXPECT_THROW(
      {
        const auto dtype = non_scalar_meta.toScalarType();
        (void)dtype;
      },
      std::exception);
}

TEST(TypeMetaCompatTest, BuiltinKnownTypeRepeatRegistrationIsStable) {
  const auto vector_meta_1 = caffe2::TypeMeta::Make<std::vector<int64_t>>();
  const auto vector_meta_2 = caffe2::TypeMeta::Make<std::vector<int64_t>>();

  EXPECT_EQ(vector_meta_1, vector_meta_2);
  EXPECT_EQ(vector_meta_1.id(), vector_meta_2.id());
}

TEST(TypeMetaCompatTest, DefaultConstructedTypeMetaIsUndefined) {
  caffe2::TypeMeta meta;

  EXPECT_TRUE(meta.isScalarType());
  EXPECT_TRUE(meta.isScalarType(c10::ScalarType::Undefined));
  EXPECT_EQ(meta.itemsize(), 0U);
  EXPECT_EQ(meta.name(), "Undefined");
  EXPECT_EQ(meta.id(),
            caffe2::TypeIdentifier::Get<caffe2::detail::_Uninitialized>());
  EXPECT_EQ(meta.toScalarType(), c10::ScalarType::Undefined);
}

TEST(TypeMetaCompatTest, AssignFromScalarTypeAndHelpersWork) {
  caffe2::TypeMeta meta;
  meta = c10::kLong;

  EXPECT_TRUE(meta.isScalarType(c10::kLong));
  EXPECT_TRUE(meta.Match<int64_t>());
  EXPECT_EQ(meta.itemsize(), sizeof(int64_t));
  EXPECT_EQ(caffe2::TypeMeta::Id<int64_t>(), meta.id());
  EXPECT_EQ(caffe2::TypeMeta::ItemSize<int64_t>(), sizeof(int64_t));
  EXPECT_FALSE(caffe2::TypeMeta::TypeName<int64_t>().empty());

  std::ostringstream stream;
  stream << meta;
  EXPECT_FALSE(stream.str().empty());
}

TEST(TypeMetaCompatTest, FundamentalTypesSkipPlacementAndCopyHooks) {
  const auto int_meta = caffe2::TypeMeta::Make<int>();
  const auto float_ptr_meta = caffe2::TypeMeta::Make<float*>();

  EXPECT_NE(int_meta.newFn(), nullptr);
  EXPECT_EQ(int_meta.placementNew(), nullptr);
  EXPECT_EQ(int_meta.copy(), nullptr);
  EXPECT_EQ(int_meta.placementDelete(), nullptr);
  EXPECT_NE(int_meta.deleteFn(), nullptr);

  EXPECT_NE(float_ptr_meta.newFn(), nullptr);
  EXPECT_EQ(float_ptr_meta.placementNew(), nullptr);
  EXPECT_EQ(float_ptr_meta.copy(), nullptr);
  EXPECT_EQ(float_ptr_meta.placementDelete(), nullptr);
  EXPECT_NE(float_ptr_meta.deleteFn(), nullptr);
}

TEST(TypeMetaCompatTest, RegisteredCustomTypeLifecycleHooksWork) {
  const auto meta = caffe2::TypeMeta::Make<LifecycleTrackedType>();

  EXPECT_TRUE(meta.Match<LifecycleTrackedType>());
  EXPECT_EQ(meta.itemsize(), sizeof(LifecycleTrackedType));
  EXPECT_NE(meta.newFn(), nullptr);
  EXPECT_NE(meta.placementNew(), nullptr);
  EXPECT_NE(meta.copy(), nullptr);
  EXPECT_NE(meta.placementDelete(), nullptr);
  EXPECT_NE(meta.deleteFn(), nullptr);
  EXPECT_FALSE(meta.name().empty());

  auto* heap_object = static_cast<LifecycleTrackedType*>(meta.newFn()());
  EXPECT_EQ(heap_object->value, 7);
  heap_object->value = 19;
  meta.deleteFn()(heap_object);

  alignas(LifecycleTrackedType) std::byte storage[sizeof(LifecycleTrackedType)];
  meta.placementNew()(storage, 1);
  auto* placed = reinterpret_cast<LifecycleTrackedType*>(storage);
  EXPECT_EQ(placed->value, 7);
  placed->value = 23;

  alignas(LifecycleTrackedType)
      std::byte copy_storage[sizeof(LifecycleTrackedType)];
  meta.placementNew()(copy_storage, 1);
  meta.copy()(storage, copy_storage, 1);
  auto* copied = reinterpret_cast<LifecycleTrackedType*>(copy_storage);
  EXPECT_EQ(copied->value, 23);

  meta.placementDelete()(copy_storage, 1);
  meta.placementDelete()(storage, 1);
}

TEST(TypeMetaCompatTest, NonDefaultConstructibleHooksThrow) {
  const auto meta = caffe2::TypeMeta::Make<NonDefaultConstructibleType>();

  EXPECT_NE(meta.newFn(), nullptr);
  EXPECT_NE(meta.placementNew(), nullptr);
  EXPECT_THROW(
      {
        auto* ptr = static_cast<NonDefaultConstructibleType*>(meta.newFn()());
        (void)ptr;
      },
      std::exception);

  alignas(NonDefaultConstructibleType)
      std::byte storage[sizeof(NonDefaultConstructibleType)];
  EXPECT_THROW(meta.placementNew()(storage, 1), std::exception);
}

TEST(TypeMetaCompatTest, NonCopyAssignableCopyHookThrows) {
  const auto meta = caffe2::TypeMeta::Make<NonCopyAssignableType>();

  EXPECT_NE(meta.copy(), nullptr);

  alignas(NonCopyAssignableType)
      std::byte src_storage[sizeof(NonCopyAssignableType)];
  alignas(NonCopyAssignableType)
      std::byte dst_storage[sizeof(NonCopyAssignableType)];
  meta.placementNew()(src_storage, 1);
  meta.placementNew()(dst_storage, 1);

  auto* src = reinterpret_cast<NonCopyAssignableType*>(src_storage);
  src->value = 31;

  EXPECT_THROW(meta.copy()(src_storage, dst_storage, 1), std::exception);

  meta.placementDelete()(dst_storage, 1);
  meta.placementDelete()(src_storage, 1);
}
