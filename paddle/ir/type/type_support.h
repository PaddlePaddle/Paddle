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

/*
 * Abstract the properties and behaviors common to all Type classes into an
 * AbstractType class. There are two types in Type system:
 * non-parameter/singleton type and parameter-type. The common attributes of all
 * types is TypeID (and possibly others). Therefore, construct a class with
 * TypeID as its member
 */
class AbstractType {
 public:
  // Construct an AbstractType by TypeID directly.
  static AbstractType Get(TypeID type_id) { return AbstractType(type_id); }

  TypeID GetTypeID() const { return typeID_; }

  // TODO(zhangbo9674): After the IRContext is designed, AbstractType will be
  // cached to IRContext with TypeID as key. static const AbstractType&
  // LookUp(IRContext* ctx, TypeID type_id);

 private:
  // The constructor is set to private and provides the user with the Get method
  // and LookUp method to obtain and manage the AstractType.
  explicit AbstractType(TypeID typeID) : typeID_(typeID) {}
  TypeID typeID_;
};

/*
 * TypeStorage is used to store all information of a Type. A Type object
 * contains a TypeStorage. For non-parameter type, the information includes:
 * TypeID, so TypeStorage only needs to include AbstractType; For parameter
 * type, in addition to AbstractType/TypeID, parameter information needs to be
 * included. So that, non-parameter type can be constructed by TypeStorage
 * directly but parameter type should be constructed by Derived TypeStorage
 */
class TypeStorage {
 public:
  const AbstractType &GetAbstractType() { return *abstract_type_; }
  TypeStorage() {}
  void Initialize(const AbstractType &abstract_type) {
    abstract_type_ = const_cast<AbstractType *>(&abstract_type);
  }

 private:
  AbstractType *abstract_type_{nullptr};
};

}  // namespace ir

// Custom specialization of std::hash can be injected in namespace std.
namespace std {
template <>
struct hash<ir::TypeID> {
  std::size_t operator()(const ir::TypeID &obj) const {
    return std::hash<const void *>()(obj.GetStorage());
  }
};
}  // namespace std
