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

#include "paddle/ir/storage_uniquer.h"
#include "paddle/ir/type_id.h"

namespace ir {
///
/// \brief A utility class, when defining a concrete type ConcreteT, you need to
/// consider its base class BaseT and the corresponding memory type StorageT. We
/// connect the three by providing a TypeBase template tool class.
///
template <typename ConcreteT,
          typename BaseT,
          typename StorageT,
          typename UniquerT>
class StorageUserBase : public BaseT {
 public:
  using BaseT::BaseT;

  using Base = StorageUserBase<ConcreteT, BaseT, StorageT, UniquerT>;

  using ImplType = StorageT;

  static TypeId type_id() { return TypeId::get<ConcreteT>(); }

  template <typename T>
  static bool classof(T val) {
    return val.type_id() == type_id();
  }

  template <typename... Args>
  static ConcreteT get(IrContext *ctx, Args... args) {
    return UniquerT::template get<ConcreteT>(ctx, args...);
  }

  /// Utility for easy access to the storage instance.
  ImplType *impl() const { return static_cast<ImplType *>(this->impl_); }
};

}  // namespace ir
