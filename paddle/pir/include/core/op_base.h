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

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/interface_support.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
class Builder;
class IrPrinter;
class Block;

class IR_API OpBase {
 public:
  explicit OpBase(Operation *operation = nullptr) : operation_(operation) {}

  Operation *operation() const {
    PADDLE_ENFORCE_NOT_NULL(
        operation_,
        phi::errors::InvalidArgument("Can't use operation() in a null op."));
    return operation_;
  }

  explicit operator bool() const { return operation_ != nullptr; }

  operator Operation *() const { return operation(); }

  Operation *operator->() const { return operation(); }

  IrContext *ir_context() const { return operation()->ir_context(); }

  uint32_t num_results() const { return operation()->num_results(); }

  uint32_t num_operands() const { return operation()->num_operands(); }

  Block *parent() const { return operation()->GetParent(); }

  // Attribute related interfaces
  const AttributeMap &attributes() const { return operation()->attributes(); }
  Attribute attribute(const std::string &key) const {
    return operation()->attribute(key);
  }
  template <typename T>
  T attribute(const std::string &key) const {
    return operation()->attribute<T>(key);
  }

  Value operand_source(uint32_t index) const {
    return operation()->operand_source(index);
  }
  Type operand_type(uint32_t index) const {
    return operation()->operand_type(index);
  }

  Value result(uint32_t index) const { return operation()->result(index); }

  template <typename T = Type>
  T result_type(uint32_t index) const {
    return operation()->result_type<T>(index);
  }

  void VerifySig() {}

  void VerifyRegion() {}

 protected:
  Operation *operation_;  // Not owned
};

///
/// \brief OpTrait
///
template <class ConcreteTrait>
class OpTraitBase : public OpBase {
 public:
  using Base = OpTraitBase<ConcreteTrait>;
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

  ///
  /// \brief Accessor for the ID of this interface.
  ///
  static TypeId GetInterfaceId() { return TypeId::get<ConcreteInterface>(); }

  ///
  /// \brief Checking if the given object defines the concrete interface.
  ///
  static bool classof(Operation *op) {
    return op->HasInterface<ConcreteInterface>();
  }

  static ConcreteInterface dyn_cast(Operation *op) {
    if (op && op->HasInterface<ConcreteInterface>()) {
      return ConcreteInterface(
          op, op->info().GetInterfaceImpl<ConcreteInterface>());
    }
    return ConcreteInterface(nullptr, nullptr);
  }
};

template <typename, typename = void>
struct VerifyTraitOrInterface {
  static void call(Operation *) {}
};

template <typename T>
struct VerifyTraitOrInterface<T,
                              decltype(T::Verify(
                                  std::declval<Operation *>()))> {
  static void call(Operation *op) { T::Verify(op); }
};

template <typename ConcreteOp, class... TraitOrInterface>
class Op : public OpBase {
 public:
  using OpBase::OpBase;

  using TraitList =
      typename detail::Filter<OpTraitBase,
                              std::tuple<TraitOrInterface...>>::Type;

  using InterfaceList =
      typename detail::Filter<OpInterfaceBase,
                              std::tuple<TraitOrInterface...>>::Type;

  // TODO(zhangbopd): Use classof
  static ConcreteOp dyn_cast(Operation *op) {
    if (op && op->info().id() == TypeId::get<ConcreteOp>()) {
      return ConcreteOp(op);
    }
    return ConcreteOp(nullptr);
  }

  static bool classof(const Operation *op) {
    return op && op->info().id() == TypeId::get<ConcreteOp>();
  }

  static std::set<InterfaceValue> interface_set() {
    return pir::detail::GetInterfaceSet<ConcreteOp, InterfaceList>();
  }

  static std::vector<TypeId> GetTraitSet() {
    return pir::detail::GetTraitSet<ConcreteOp, TraitList>();
  }

  // Checking that the derived class does not define any member by comparing
  // its size to an ad-hoc EmptyOp.
  static constexpr bool HasNoDataMembers() {
    class EmptyOp : public Op<EmptyOp, TraitOrInterface...> {};
    return sizeof(ConcreteOp) == sizeof(EmptyOp);
  }

  // Implementation of `VerifySigInvariantsFn` OperationName hook.
  static void VerifySigInvariants(Operation *op) {
    static_assert(HasNoDataMembers(),
                  "Op class shouldn't define new data members");
    op->dyn_cast<ConcreteOp>().VerifySig();
    (void)std::initializer_list<int>{
        0, (VerifyTraitOrInterface<TraitOrInterface>::call(op), 0)...};
  }

  static void VerifyRegionInvariants(Operation *op) {
    static_assert(HasNoDataMembers(),
                  "Op class shouldn't define new data members");
    op->dyn_cast<ConcreteOp>().VerifyRegion();
  }
};

}  // namespace pir
