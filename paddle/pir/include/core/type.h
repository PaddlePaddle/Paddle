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

#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/storage_manager_support.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/type_id.h"

namespace pir {
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
            typename BaseType,
            typename StorageType,
            class... TraitOrInterface>
  using TypeBase = detail::StorageHelperBase<ConcreteType,
                                             BaseType,
                                             StorageType,
                                             TypeManager,
                                             TraitOrInterface...>;
  using Storage = TypeStorage;
  using AbstractT = AbstractType;

  Type() = default;

  Type(const Storage *storage) : storage_(storage) {}  // NOLINT

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

  ///
  /// \brief Support PointerLikeTypeTraits.
  ///
  operator const void *() const { return storage_; }
  static Type RecoverFromVoidPointer(const void *pointer) {
    return Type(reinterpret_cast<const Storage *>(pointer));
  }

  ///
  /// \brief Return the abstract type descriptor.
  ///
  const AbstractT &abstract_type();

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
    return *this && pir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return CastUtil<U>::call(*this);
  }

  void Print(std::ostream &os) const;

  static Type Parse(std::istream &is, IrContext *ctx);

  template <typename U>
  U cast() const {
    return pir::cast<U>(*this);
  }

  static Type dyn_cast_impl(Type type) { return type; }

  ///
  /// \brief Return true if this is an integer (any signedness) or an index
  /// type.
  ///
  bool IsIntOrIndex() const;
  bool IsIndex() const;

  std::size_t hash() const { return std::hash<const void *>()(storage_); }

 protected:
  const Storage *storage_{nullptr};

 private:
  template <typename To, typename Enabler = void>
  struct CastUtil {
    static To call(Type type) {
      throw("Can't dyn_cast to To, To should be a Type or Interface or Trait");
    }
  };

  template <typename To>
  struct CastUtil<
      To,
      typename std::enable_if<std::is_base_of<Type, To>::value>::type> {
    static inline To call(Type type) { return To::dyn_cast_impl(type); }
  };
};

IR_API std::ostream &operator<<(std::ostream &os, Type type);

///
/// \brief This class represents the base of a type interface.
///
template <typename ConcreteInterface>
class TypeInterfaceBase : public pir::Type {
 public:
  explicit TypeInterfaceBase(Type type) : Type(type) {}

  ///
  /// \brief Accessor for the ID of this interface.
  ///
  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }

  ///
  /// \brief Checking if the given object defines the concrete interface.
  ///
  static bool classof(Type type) {
    return type.abstract_type().HasInterface(TypeId::get<ConcreteInterface>());
  }

  static ConcreteInterface dyn_cast_impl(Type type) {
    if (type &&
        type.abstract_type().HasInterface(TypeId::get<ConcreteInterface>())) {
      return ConcreteInterface(
          type, type.abstract_type().GetInterfaceImpl<ConcreteInterface>());
    }
    return ConcreteInterface(nullptr, nullptr);
  }
};

}  // namespace pir

namespace std {
///
/// \brief Enable hashing Type.
///
template <>
struct hash<pir::Type> {
  std::size_t operator()(const pir::Type &obj) const { return obj.hash(); }
};
}  // namespace std
