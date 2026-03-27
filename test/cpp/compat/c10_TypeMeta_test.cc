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
#include <c10/util/typeid.h>

#include <array>
#include <exception>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

caffe2::TypeIdentifier GetIntTypeIdentifierFromHelper() {
  return caffe2::TypeIdentifier::Get<int>();
}

caffe2::TypeMeta MakeStdStringTypeMetaFromHelper() {
  return caffe2::TypeMeta::Make<std::string>();
}

}  // namespace

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
