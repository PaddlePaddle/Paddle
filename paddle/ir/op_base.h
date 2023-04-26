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

#include "paddle/ir/operation.h"
#include "paddle/ir/utils.h"

namespace ir {
class OpBase {
 public:
  explicit OpBase(const Operation *operation) : operation_(operation) {}

  const Operation *operation() const { return operation_; }

  explicit operator bool() const { return operation() != nullptr; }

  operator const Operation *() const { return operation_; }

  const Operation *operator->() const { return operation_; }

 private:
  const Operation *operation_;  // Not owned
};

///
/// \brief OpTrait
///
template <class ConcreteTrait>
class OpTraitBase : public OpBase {
 public:
  explicit OpTraitBase(const Operation *op) : OpBase(op) {}

  static TypeId GetTraitId() { return TypeId::get<ConcreteTrait>(); }
};

///
/// \brief OpInterface
///
template <typename ConcreteInterface>
class OpInterfaceBase : public OpBase {
 public:
  // explicit OpInterfaceBase(Operation *op) : OpBase(op) {}

  explicit OpInterfaceBase(const Operation *op) : OpBase(op) {}

  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }
};

template <typename ConcreteOp, class... TraitOrInterface>
class Op : public OpBase {
 public:
  using OpBase::OpBase;

  using TraitList =
      typename Filter<OpTraitBase, std::tuple<TraitOrInterface...>>::Type;

  using InterfaceList =
      typename Filter<OpInterfaceBase, std::tuple<TraitOrInterface...>>::Type;
};

}  // namespace ir
