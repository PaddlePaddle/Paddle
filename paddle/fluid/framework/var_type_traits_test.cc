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
#include <unordered_set>

namespace paddle {
namespace framework {

template <int kPos, int kEnd, bool kStop>
struct TypeIndexChecker {
  template <typename SetType1, typename SetType2>
  static void Check(SetType1 *var_id_set, SetType2 *type_index_set) {
    using Type =
        typename std::tuple_element<kPos, VarTypeRegistry::ArgTuple>::type;
    static_assert(std::is_same<typename VarTypeTrait<Type>::Type, Type>::value,
                  "Type must be the same");
    constexpr auto kId = VarTypeTrait<Type>::kId;
    if (!std::is_same<Type, void>::value) {
      std::type_index actual_type(typeid(Type));
      EXPECT_EQ(std::string(ToTypeName(kId)), std::string(actual_type.name()));
      EXPECT_EQ(ToTypeIndex(kId), actual_type);
      EXPECT_TRUE(var_id_set->count(kId) == 0);              // NOLINT
      EXPECT_TRUE(type_index_set->count(actual_type) == 0);  // NOLINT
      var_id_set->insert(kId);
      type_index_set->insert(std::type_index(typeid(Type)));
    }
    TypeIndexChecker<kPos + 1, kEnd, kPos + 1 == kEnd>::Check(var_id_set,
                                                              type_index_set);
  }
};

template <int kPos, int kEnd>
struct TypeIndexChecker<kPos, kEnd, true> {
  template <typename SetType1, typename SetType2>
  static void Check(SetType1 *, SetType2 *) {}
};

TEST(var_type_traits, check_no_duplicate_registry) {
  constexpr size_t kRegisteredNum = VarTypeRegistry::kRegisteredTypeNum;
  std::unordered_set<int> var_id_set;
  std::unordered_set<std::type_index> type_index_set;
  TypeIndexChecker<0, kRegisteredNum, kRegisteredNum == 0>::Check(
      &var_id_set, &type_index_set);
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
