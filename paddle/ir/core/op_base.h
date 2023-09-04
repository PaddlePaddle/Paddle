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
#include <type_traits>

#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/interface_support.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/utils.h"

namespace ir {

class IR_API OpBase {
 public:
  explicit OpBase(Operation *operation = nullptr) : operation_(operation) {}

  Operation *operation() const {
    IR_ENFORCE(operation_, "Can't use operation() in a null op.");
    return operation_;
  }

  explicit operator bool() const { return operation_ != nullptr; }

  operator Operation *() const { return operation(); }

  Operation *operator->() const { return operation(); }

  IrContext *ir_context() const { return operation()->ir_context(); }

  uint32_t num_results() const { return operation()->num_results(); }

  uint32_t num_operands() const { return operation()->num_operands(); }

  const AttributeMap &attributes() const { return operation()->attributes(); }

  Value operand_source(uint32_t index) const {
    return operation()->operand_source(index);
  }

  OpResult result(uint32_t index) const { return operation()->result(index); }

  ir::Attribute attribute(const std::string &name) {
    return operation()->attribute(name);
  }

  template <typename T>
  T attribute(const std::string &name) {
    return operation()->attribute<T>(name);
  }

 private:
  Operation *operation_;  // Not owned
};

///
/// \brief OpTrait
///
template <class ConcreteTrait>
class OpTraitBase : public OpBase {
 public:
  explicit OpTraitBase(Operation *op) : OpBase(op) {}

  static TypeId GetTraitId() { return TypeId::get<ConcreteTrait>(); }

  static ConcreteTrait dyn_cast(Operation *op) {
    if (op && op->HasTrait<ConcreteTrait>()) {
      return ConcreteTrait(op);
    }
    return ConcreteTrait(nullptr);
  }
};

///
/// \brief OpInterface
///
template <typename ConcreteInterface>
class OpInterfaceBase : public OpBase {
 public:
  explicit OpInterfaceBase(Operation *op) : OpBase(op) {}

  // Accessor for the ID of this interface.
  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }

  static ConcreteInterface dyn_cast(Operation *op) {
    if (op && op->HasInterface<ConcreteInterface>()) {
      return ConcreteInterface(
          op, op->info().GetInterfaceImpl<ConcreteInterface>());
    }
    return ConcreteInterface(nullptr, nullptr);
  }
};

template <typename ConcreteOp, class... TraitOrInterface>
class Op : public OpBase {
 public:
  using OpBase::OpBase;

  using TraitList =
      typename Filter<OpTraitBase, std::tuple<TraitOrInterface...>>::Type;

  using InterfaceList =
      typename Filter<OpInterfaceBase, std::tuple<TraitOrInterface...>>::Type;

  static ConcreteOp dyn_cast(Operation *op) {
    if (op && op->info().id() == TypeId::get<ConcreteOp>()) {
      return ConcreteOp(op);
    }
    return ConcreteOp(nullptr);
  }

  // Returns an interface map containing the interfaces registered to this
  // operation.
  static std::vector<details::InterfaceValue> GetInterfaceMap() {
    constexpr size_t interfaces_num = std::tuple_size<InterfaceList>::value;
    std::vector<details::InterfaceValue> interfaces_map(interfaces_num);
    ConstructInterfacesOrTraits<ConcreteOp, InterfaceList>::interface(
        interfaces_map.data());
    return interfaces_map;
  }

  static std::vector<TypeId> GetTraitSet() {
    constexpr size_t traits_num = std::tuple_size<TraitList>::value;
    std::vector<TypeId> trait_set(traits_num);
    auto p_first_trait = trait_set.data();
    ConstructInterfacesOrTraits<ConcreteOp, TraitList>::trait(p_first_trait);
    return trait_set;
  }

  // Checking that the derived class does not define any member by comparing
  // its size to an ad-hoc EmptyOp.
  static constexpr bool HasNoDataMembers() {
    class EmptyOp : public Op<EmptyOp, TraitOrInterface...> {};
    return sizeof(ConcreteOp) == sizeof(EmptyOp);
  }

  // Implementation of `VerifyInvariantsFn` OperationName hook.
  static void VerifyInvariants(Operation *op) {
    static_assert(HasNoDataMembers(),
                  "Op class shouldn't define new data members");
    op->dyn_cast<ConcreteOp>().Verify();
  }
};

}  // namespace ir
