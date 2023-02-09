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

/// \brief TypeId is the unique identification of Type, each Type corresponds to
/// a unique TypeId, the same id indicates the same Type class. TypeId provides
/// an instantiation interface: TypeId::get.
/// Example:
/// \code{cpp}
///   class TypeA {};
///   TypeId type_a_id = TypeId::get<TypeA>();
/// \endcode
class TypeId {
  struct Storage {};

 public:
  /// \brief Returns the unique TypeId of Type T.
  /// \return The unique TypeId of Type T.
  template <typename T>
  static TypeId get() {
    static Storage instance;
    return TypeId(&instance);
  }

  /// \brief Comparison operations.
  inline bool operator==(const TypeId &other) const {
    return storage_ == other.storage_;
  }

  /// \brief Comparison operations.
  inline bool operator!=(const TypeId &other) const {
    return !(*this == other);
  }

  /// \brief Enable hashing TypeId instances.
  friend struct std::hash<TypeId>;

 private:
  /// \brief Construct a TypeId and initialize storage.
  /// \param storage The storage of this TypeId.
  explicit TypeId(const Storage *storage) : storage_(storage) {}

  const Storage *storage_;
};

/// \brief Abstract the properties and behaviors common to all Type classes into
/// an AbstractType class. There are two types in Type system:
/// on-parameter/singleton type and parameter-type. The common attributes of all
/// types is TypeId (and possibly others). Therefore, construct a class with
/// TypeId as its member.
class AbstractType {
 public:
  /// \brief Construct an AbstractType by TypeId directly.
  /// \param type_id The type id of the AbstractType.
  static AbstractType get(TypeId type_id) { return AbstractType(type_id); }

  /// \brief Returns the type id of the AbstractType.
  /// \return The type id of the AbstractType.
  TypeId type_id() const { return type_id_; }

  /* TODO(zhangbo9674): After the IRContext is designed, AbstractType will be
   * cached to IRContext with TypeId as key.
   */

 private:
  /// \brief The constructor is set to private and provides the user with the
  /// get method to obtain and manage the AstractType.
  /// \param type_id The type id of the AbstractType.
  explicit AbstractType(TypeId type_id) : type_id_(type_id) {}

  TypeId type_id_;
};

/// \brief TypeStorage is used to store all information of a Type. A Type object
/// contains a TypeStorage. For non-parameter type, the information includes:
/// TypeId, so TypeStorage only needs to include AbstractType; For parameter
/// type, in addition to AbstractType/TypeId, parameter information needs to be
/// included. So that, non-parameter type can be constructed by TypeStorage
/// directly but parameter type should be constructed by Derived TypeStorage.
class TypeStorage {
 public:
  /// \brief Construct a TypeStorage and initialize abstract_type.
  /// \param abstract_type The abstract_type of this TypeStorage.
  explicit TypeStorage(AbstractType *abstract_type)
      : abstract_type_(abstract_type) {}

  /// \brief Returns the AbstractType of the TypeStorage.
  /// \return The AbstractType of the TypeStorage.
  const AbstractType &abstract_type() { return *abstract_type_; }

 private:
  AbstractType *abstract_type_{nullptr};
};

}  // namespace ir

// Custom specialization of std::hash can be injected in namespace std.
namespace std {
/// \brief Enable hashing TypeId instances.
template <>
struct hash<ir::TypeId> {
  std::size_t operator()(const ir::TypeId &obj) const {
    return std::hash<const ir::TypeId::Storage *>()(obj.storage_);
  }
};
}  // namespace std
