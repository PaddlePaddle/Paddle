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

namespace paddle {
namespace framework {

// Besides registering variable type id, it is helpful to register a
// var_id -> std::type_index map (for example, get type names according to id)
namespace detail {

template <int kStart, int kEnd, bool kStop>
struct VarIdToTypeIndexMapInitializerImpl {
  static void Init(std::unordered_map<int, std::type_index> *m) {
    using Type =
        typename std::tuple_element<kStart, VarTypeRegistry::ArgTuple>::type;
    constexpr int kId = VarTypeTrait<Type>::kId;
    if (!std::is_same<Type, void>::value) {
      m->emplace(kId, std::type_index(typeid(Type)));
    }
    VarIdToTypeIndexMapInitializerImpl<kStart + 1, kEnd,
                                       kStart + 1 == kEnd>::Init(m);
  }
};

template <int kStart, int kEnd>
struct VarIdToTypeIndexMapInitializerImpl<kStart, kEnd, true> {
  static void Init(std::unordered_map<int, std::type_index> *m) {}
};

// VarIdToTypeIndexMapInitializer is designed to initialize var_id ->
// std::type_index map
using VarIdToTypeIndexMapInitializer =
    VarIdToTypeIndexMapInitializerImpl<0, VarTypeRegistry::kRegisteredTypeNum,
                                       VarTypeRegistry::kRegisteredTypeNum ==
                                           0>;

struct VarIdToTypeIndexMapHolder {
 public:
  static const std::type_index &ToTypeIndex(int var_id) {
    static const VarIdToTypeIndexMapHolder instance;
    auto it = instance.var_type_map_.find(var_id);
    PADDLE_ENFORCE(it != instance.var_type_map_.end(),
                   "VarId %d is not registered.", var_id);
    return it->second;
  }

 private:
  VarIdToTypeIndexMapHolder() {
    VarIdToTypeIndexMapInitializer::Init(&var_type_map_);
  }
  std::unordered_map<int, std::type_index> var_type_map_;
};

}  // namespace detail

const char *ToTypeName(int var_id) { return ToTypeIndex(var_id).name(); }

const std::type_index &ToTypeIndex(int var_id) {
  return detail::VarIdToTypeIndexMapHolder::ToTypeIndex(var_id);
}

}  // namespace framework
}  // namespace paddle
