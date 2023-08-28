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

#pragma once

#include <ostream>

#include "paddle/ir/core/cast_utils.h"
#include "paddle/ir/core/storage_manager_support.h"
#include "paddle/ir/core/type_id.h"
namespace ir {
class TypeStorage;
class AbstractType;
class IrContext;
class Dialect;
///
/// \brief Unified interface of the Type class. Derivation of all Type classes
/// only derives interfaces, not members. For example, DenseTensorType,
/// Float32Type, etc. are all derived classes of Type, but no new member
/// variables will be added.
///
class IR_API Type {
 public:
  template <typename ConcreteType,
            typename StorageType,
            class... TraitOrInterface>  // Traits or Interface
  using TypeBase = detail::StorageHelperBase<ConcreteType,
                                             StorageType,
                                             ir::TypeManager,
                                             TraitOrInterface...>;

  using Storage = TypeStorage;

  using AbstractT = AbstractType;

  Type() = default;

  Type(const Storage *storage)  // NOLINT
      : storage_(const_cast<Storage *>(storage)) {}

  Type(const Type &other) = default;

  Type &operator=(const Type &other) = default;

  ///
  /// \brief Some operators are overloaded.
  ///
  bool operator==(Type other) const { return storage_ == other.storage_; }

  bool operator!=(Type other) const { return storage_ != other.storage_; }

  explicit operator bool() const { return storage_; }

  bool operator!() const { return storage_ == nullptr; }

  ///
  /// \brief Some type attribute acquisition interfaces.
  ///
  TypeId type_id();

  const AbstractType &abstract_type();

  ///
  /// \brief Return the Type implementation.
  ///
  const Storage *storage() const { return storage_; }

  Dialect &dialect() const;

  IrContext *ir_context() const;

  ///
  /// \brief Methods for type judgment and cast.
  ///
  static bool classof(Type) { return true; }

  template <typename T>
  bool isa() const {
    return ir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return ir::dyn_cast<U>(*this);
  }

  void Print(std::ostream &os) const;

  ///
  /// \brief Enable hashing Type.
  ///
  friend struct std::hash<Type>;

 protected:
  const Storage *storage_{nullptr};
};

IR_API std::ostream &operator<<(std::ostream &os, Type type);

///
/// \brief This class represents the base of a type interface.
///

template <typename ConcreteType>
class TypeInterfaceBase : public Type {
 public:
  TypeInterfaceBase() : Type() {}
  explicit TypeInterfaceBase(Type type) : Type() {}
  static ConcreteType dyn_cast(Type *type);
};

}  // namespace ir

namespace std {
///
/// \brief Enable hashing Type.
///
template <>
struct hash<ir::Type> {
  std::size_t operator()(const ir::Type &obj) const {
    return std::hash<const ir::Type::Storage *>()(obj.storage_);
  }
};
}  // namespace std
