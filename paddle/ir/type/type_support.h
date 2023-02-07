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

#include <functional>

namespace ir {

/*
 * TypeId is the unique identification of Type, each Type corresponds to a
 * unique TypeId, the same id indicates the same Type class. TypeId provides an
 * instantiation interface: TypeId::get.
 * For example:
 *   class TypeA {};
 *   TypeId type_a_id = TypeId::get<TypeA>();
 */
class TypeId {
  struct Storage {};

 public:
  template <typename T>
  static TypeId get() {
    static Storage instance;
    return TypeId(&instance);
  }
  inline bool operator==(const TypeId &other) const {
    return storage_ == other.storage_;
  }
  inline bool operator!=(const TypeId &other) const {
    return !(*this == other);
  }
  friend struct std::hash<TypeId>;

 private:
  explicit TypeId(const Storage *storage) : storage_(storage) {}
  const Storage *storage_;
};

/*
 * Abstract the properties and behaviors common to all Type classes into an
 * AbstractType class. There are two types in Type system:
 * non-parameter/singleton type and parameter-type. The common attributes of all
 * types is TypeId (and possibly others). Therefore, construct a class with
 * TypeId as its member
 */
class AbstractType {
 public:
  // Construct an AbstractType by TypeId directly.
  static AbstractType get(TypeId type_id) { return AbstractType(type_id); }

  TypeId type_id() const { return type_id_; }

  // TODO(zhangbo9674): After the IRContext is designed, AbstractType will be
  // cached to IRContext with TypeId as key. static const AbstractType&
  // LookUp(IRContext* ctx, TypeId type_id);

 private:
  // The constructor is set to private and provides the user with the Get method
  // and LookUp method to obtain and manage the AstractType.
  explicit AbstractType(TypeId type_id) : type_id_(type_id) {}
  TypeId type_id_;
};

/*
 * TypeStorage is used to store all information of a Type. A Type object
 * contains a TypeStorage. For non-parameter type, the information includes:
 * TypeId, so TypeStorage only needs to include AbstractType; For parameter
 * type, in addition to AbstractType/TypeId, parameter information needs to be
 * included. So that, non-parameter type can be constructed by TypeStorage
 * directly but parameter type should be constructed by Derived TypeStorage
 */
class TypeStorage {
 public:
  explicit TypeStorage(AbstractType *abstract_type)
      : abstract_type_(abstract_type) {}

  const AbstractType &abstract_type() { return *abstract_type_; }

 private:
  AbstractType *abstract_type_{nullptr};
};

}  // namespace ir

// Custom specialization of std::hash can be injected in namespace std.
namespace std {
template <>
struct hash<ir::TypeId> {
  std::size_t operator()(const ir::TypeId &obj) const {
    return std::hash<const void *>()(static_cast<const void *>(obj.storage_));
  }
};
}  // namespace std
