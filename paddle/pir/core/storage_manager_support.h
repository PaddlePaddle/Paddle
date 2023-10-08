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

#include "paddle/pir/core/interface_support.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/type_base.h"
#include "paddle/pir/core/type_id.h"

namespace pir {
template <typename ConcreteInterface>
class TypeInterfaceBase;

namespace detail {

namespace storage_helper_base_impl {
///
/// \brief Returns true if this given trait id matches the ids of any of the
/// provided trait.
///
template <template <typename T> class... Traits>
bool hasTrait(TypeId traitID) {
  TypeId traitIDs[] = {TypeId::get<Traits>()...};
  for (unsigned i = 0, e = sizeof...(Traits); i != e; ++i)
    if (traitIDs[i] == traitID) return true;
  return false;
}

// Specialize for the empty case.
inline bool hasTrait(TypeId traitID) { return false; }
}  // namespace storage_helper_base_impl

// Implementing users of storage classes uniqued by StorageManager.
template <typename ConcreteT,
          typename BaseT,
          typename StorageT,
          typename ManagerT,
          class... TraitOrInterface>
class StorageHelperBase : public BaseT {
 public:
  using BaseT::BaseT;

  using Base = StorageHelperBase<ConcreteT,
                                 BaseT,
                                 StorageT,
                                 ManagerT,
                                 TraitOrInterface...>;
  using HasTraitFn = bool (*)(TypeId);
  using Storage = StorageT;
  using InterfaceList =
      typename Filter<TypeInterfaceBase, std::tuple<TraitOrInterface...>>::Type;

  static ConcreteT dyn_cast_impl(BaseT type) {
    if (type && type.abstract_type().type_id() == TypeId::get<ConcreteT>()) {
      return ConcreteT(type.storage());
    }
    return ConcreteT(nullptr);
  }

  ///
  /// \brief Access to the storage instance.
  ///
  const Storage *storage() const {
    return static_cast<const Storage *>(this->storage_);
  }

  ///
  /// \brief Get the identifier for the concrete type.
  ///
  static pir::TypeId type_id() { return pir::TypeId::get<ConcreteT>(); }

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  template <typename T>
  static bool classof(T val) {
    return val.type_id() == type_id();
  }

  ///
  /// \brief Returns an interface map for the interfaces registered to this
  /// storage user.
  ///
  static std::vector<InterfaceValue> interface_map() {
    return pir::detail::GetInterfaceMap<ConcreteT, InterfaceList>();
  }

  ///
  /// \brief Get or create a new ConcreteT instance within the ctx.
  ///
  template <typename... Args>
  static ConcreteT get(pir::IrContext *ctx, Args... args) {
    return ManagerT::template get<ConcreteT>(ctx, args...);
  }

  ///
  /// \brief Returns the function that returns true if the given trait id
  /// matches the ids of any of the traits defined by the storage user.
  ///
  static HasTraitFn getHasTraitFn() {
    return [](TypeId id) {
      return storage_helper_base_impl::hasTrait<TraitOrInterface...>(id);
    };
  }
};
}  // namespace detail
}  // namespace pir
