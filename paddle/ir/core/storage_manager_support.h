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

#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/type_base.h"
#include "paddle/ir/core/type_id.h"

namespace ir {
class InFlightDiagnostic;
class Location;
class MLIRContext;

namespace detail {

namespace storage_helper_base_impl {
/// Returns true if this given Trait ID matches the IDs of any of the provided
/// trait types `Traits`.
template <template <typename T> class... Traits>
bool hasTrait(TypeId traitID) {
  TypeId traitIDs[] = {TypeId::get<Traits>()...};
  for (unsigned i = 0, e = sizeof...(Traits); i != e; ++i)
    if (traitIDs[i] == traitID) return true;
  return false;
}

// We specialize for the empty case to not define an empty array.
template <>
inline bool hasTrait(TypeId traitID) {
  return false;
}
}  // namespace storage_helper_base_impl

///
/// \brief Implementing users of storage classes uniqued by StorageManager.
///

template <typename ConcreteT,
          typename BaseT,
          typename StorageT,
          typename ManagerT,
          class... TraitOrInterface>  // Traits or Interface
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

  /// Utility for easy access to the storage instance.
  const Storage *storage() const {
    return static_cast<const Storage *>(this->storage_);
    return static_cast<const Storage *>(this->storage_);
  }

  /// Get the identifier for the concrete type.
  static ir::TypeId type_id() { return ir::TypeId::get<ConcreteT>(); }

  /// Provide an implementation of 'classof' that compares the type id of the
  /// provided value with that of the concrete type.
  template <typename T>
  static bool classof(T val) {
    return val.type_id() == type_id();
  }

  /// Get or create a new ConcreteT instance within the ctx.
  template <typename... Args>
  static ConcreteT get(ir::IrContext *ctx, Args... args) {
    return ManagerT::template get<ConcreteT>(ctx, args...);
  }

  /// Returns the function that returns true if the given Trait ID matches the
  /// IDs of any of the traits defined by the storage user.
  static HasTraitFn getHasTraitFn() {
    return [](TypeId id) {
      return storage_helper_base_impl::hasTrait<TraitOrInterface...>(
          id);  // 需要 filter
    };
  }
};
}  // namespace detail
}  // namespace ir
