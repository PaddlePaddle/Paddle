// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/var_type_traits.h"
#include <gtest/gtest.h>
#include <cstdint>

namespace paddle {
namespace framework {

template <int kPos, int kEnd, bool kStop>
struct TypeIndexChecker {
  static void Check() {
    using Type =
        typename std::tuple_element<kPos, VarTypeRegistry::ArgTuple>::type;
    if (!std::is_same<Type, void>::value) {
      EXPECT_TRUE(ToTypeIndex(VarTypeTrait<Type>::kId) == typeid(Type));
      EXPECT_TRUE(std::string(ToTypeName(VarTypeTrait<Type>::kId)) ==
                  typeid(Type).name());
    }
    TypeIndexChecker<kPos + 1, kEnd, kPos + 1 == kEnd>::Check();
  }
};

template <int kPos, int kEnd>
struct TypeIndexChecker<kPos, kEnd, true> {
  static void Check() {}
};

TEST(var_type_traits, check_type_index) {
  constexpr size_t kRegisteredNum = VarTypeRegistry::kRegisteredTypeNum;
  TypeIndexChecker<0, kRegisteredNum, kRegisteredNum == 0>::Check();
}

template <typename T>
bool CheckVarId(int proto_id) {
  static_assert(std::is_same<typename VarTypeTrait<T>::Type, T>::value,
                "Type must be the same");
  return VarTypeTrait<T>::kId == proto_id;
}

TEST(var_type_traits, check_proto_type_id) {
  ASSERT_TRUE(CheckVarId<LoDTensor>(proto::VarType::LOD_TENSOR));
  ASSERT_TRUE(CheckVarId<SelectedRows>(proto::VarType::SELECTED_ROWS));
  ASSERT_TRUE(CheckVarId<std::vector<Scope *>>(proto::VarType::STEP_SCOPES));
  ASSERT_TRUE(CheckVarId<LoDRankTable>(proto::VarType::LOD_RANK_TABLE));
  ASSERT_TRUE(CheckVarId<LoDTensorArray>(proto::VarType::LOD_TENSOR_ARRAY));
  ASSERT_TRUE(CheckVarId<platform::PlaceList>(proto::VarType::PLACE_LIST));
  ASSERT_TRUE(CheckVarId<ReaderHolder>(proto::VarType::READER));
}

TEST(var_type_traits, test_registry) {
  using Registry =
      detail::VarTypeRegistryImpl<int8_t, int32_t, size_t, double, void>;
  ASSERT_TRUE(Registry::TypePos<int8_t>() == 0);
  ASSERT_TRUE(Registry::TypePos<int32_t>() == 1);
  ASSERT_TRUE(Registry::TypePos<size_t>() == 2);
  ASSERT_TRUE(Registry::TypePos<double>() == 3);
  ASSERT_TRUE(Registry::TypePos<void>() == -1);
  ASSERT_TRUE(Registry::TypePos<float>() == -1);
}

}  // namespace framework
}  // namespace paddle
