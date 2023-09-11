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

#include "paddle/pir/core/cast_utils.h"
#include "paddle/pir/core/storage_manager_support.h"
#include "paddle/pir/core/type_id.h"
namespace pir {
class TypeStorage;
class AbstractType;
class IrContext;
class Dialect;
class ShapedTypeInterface;
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

  ///
  /// \brief Support PointerLikeTypeTraits.
  ///
  ///
  const void *AsOpaquePointer() const {
    return static_cast<const void *>(storage_);
  }
  static Type RecoverFromOpaquePointer(const void *pointer) {
    return Type(reinterpret_cast<Storage *>(const_cast<void *>(pointer)));
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
    return pir::isa<T>(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return pir::dyn_cast<U>(*this);
  }

  template <typename U>
  U dyn_cast_interface() const {
    return CastInfo<U>::call(*this);
  }

  void Print(std::ostream &os) const;

  static Type Parse(std::istream &is, IrContext *ctx);

  ///
  /// \brief Enable hashing Type.
  ///
  friend struct std::hash<Type>;

  template <typename U>
  U cast() const {
    return ir::cast<U>(*this);
  }

 protected:
  const Storage *storage_{nullptr};

 private:
  template <typename T, typename Enabler = void>
  struct CastInfo {
    static T call(Type type) {
      throw("Can't dyn_cast to T, T should be a Type or Interface");
    }
  };

  template <typename T>
  struct CastInfo<
      T,
      typename std::enable_if<std::is_base_of<ir::Type, T>::value>::type> {
    static inline T call(ir::Type type) { return T::dyn_cast(type); }
  };
};

IR_API std::ostream &operator<<(std::ostream &os, Type type);

///
/// \brief This class represents the base of a type interface.
///
template <typename ConcreteInterface>
class TypeInterfaceBase : public ir::Type {
 public:
  explicit TypeInterfaceBase(Type type) : Type(type) {}

  // Accessor for the ID of this interface.
  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }

  static ConcreteInterface dyn_cast(Type type) {
    return ConcreteInterface(
        type, type.abstract_type().GetInterfaceImpl<ConcreteInterface>());
  }
};

template <typename To, typename From>
struct cast_impl<
    To,
    From,
    typename std::enable_if<std::is_base_of<ir::Type, From>::value>::type> {
  static inline To call(const ir::Type type) { return To(type.storage()); }
};

}  // namespace pir

namespace std {
///
/// \brief Enable hashing Type.
///
template <>
struct hash<pir::Type> {
  std::size_t operator()(const pir::Type &obj) const {
    return std::hash<const pir::Type::Storage *>()(obj.storage_);
  }
};
}  // namespace std
