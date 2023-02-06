//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <iostream>

namespace paddle {
namespace framework {
namespace ir {

/*
 * TypeID is the unique identification of Type, each Type corresponds to a
 * unique TypeID, the same ID indicates the same Type class. TypeID provides an
 * instantiation interface: TypeID::Get.
 * For example:
 *   class Type_A {};
 *   TypeID type_a_id = TypeID::Get<Type_A>();
 *   void* type_a_storage = type_a_id.GetStorage();
 */
class TypeID {
  struct Storage {};

 public:
  template <typename T>
  static TypeID Get() {
    static Storage instance;
    return TypeID(&instance);
  }
  static TypeID Get(const void *ptr) {
    return TypeID(reinterpret_cast<const Storage *>(ptr));
  }
  const void *GetStorage() const { return static_cast<const void *>(storage_); }
  inline bool operator==(const TypeID &other) const {
    return storage_ == other.storage_;
  }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

 private:
  explicit TypeID(const Storage *storage) : storage_(storage) {}
  const Storage *storage_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

// Custom specialization of std::hash can be injected in namespace std.
namespace std {
template <>
struct hash<paddle::framework::ir::TypeID> {
  std::size_t operator()(const paddle::framework::ir::TypeID &obj) const {
    return std::hash<const void *>()(obj.GetStorage());
  }
};
}  // namespace std
