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

#include "paddle/ir/operation.h"
#include "paddle/ir/utils.h"

namespace ir {

class InterfaceValue {
 public:
  template <typename ConcreteOp, typename T>
  static InterfaceValue get() {
    InterfaceValue val;
    val.type_id_ = TypeId::get<T>();
    val.model_ = malloc(sizeof(typename T::template Model<ConcreteOp>));
    if (val.model_ == nullptr) {
      throw("Alloc memory for interface failed.");
    }
    static_assert(std::is_trivially_destructible<
                      typename T::template Model<ConcreteOp>>::value,
                  "interface models must be trivially destructible");
    new (val.model_) typename T::template Model<ConcreteOp>();
    return val;
  }
  TypeId type_id() const { return type_id_; }
  void *model() const { return model_; }

  InterfaceValue() = default;
  explicit InterfaceValue(TypeId type_id) : type_id_(type_id) {}
  InterfaceValue(const InterfaceValue &) = delete;
  InterfaceValue(InterfaceValue &&);
  InterfaceValue &operator=(const InterfaceValue &) = delete;
  InterfaceValue &operator=(InterfaceValue &&);
  ~InterfaceValue();
  void swap(InterfaceValue &&val) {
    using std::swap;
    swap(type_id_, val.type_id_);
    swap(model_, val.model_);
  }

  ///
  /// \brief Comparison operations.
  ///
  inline bool operator<(const InterfaceValue &other) const {
    return type_id_ < other.type_id_;
  }

 private:
  TypeId type_id_;
  void *model_{nullptr};
};

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

template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits {
 public:
  /// Construct method for interfaces.
  static InterfaceValue *interface(InterfaceValue *p_interface) {
    (void)std::initializer_list<int>{
        0, (PlacementConstrctInterface<Args>(p_interface), 0)...};
    return p_interface;
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    (void)std::initializer_list<int>{
        0, (PlacementConstrctTrait<Args>(p_trait), 0)...};
    return p_trait;
  }

 private:
  /// Placement new interface.
  template <typename T>
  static void PlacementConstrctInterface(
      InterfaceValue *&p_interface) {  // NOLINT
    p_interface->swap(InterfaceValue::get<ConcreteOp, T>());
    VLOG(4) << "New a interface: id[" << (p_interface->type_id()).storage()
            << "].";
    ++p_interface;
  }

  /// Placement new trait.
  template <typename T>
  static void PlacementConstrctTrait(ir::TypeId *&p_trait) {  // NOLINT
    *p_trait = TypeId::get<T>();
    VLOG(4) << "New a trait: id[" << p_trait->storage() << "].";
    ++p_trait;
  }
};

/// Specialized for tuple type.
template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits<ConcreteOp, std::tuple<Args...>> {
 public:
  /// Construct method for interfaces.
  static InterfaceValue *interface(InterfaceValue *p_interface) {
    return ConstructInterfacesOrTraits<ConcreteOp, Args...>::interface(
        p_interface);
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    return ConstructInterfacesOrTraits<ConcreteOp, Args...>::trait(p_trait);
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

  static std::vector<InterfaceValue> GetInterfaceMap() {
    constexpr size_t interfaces_num = std::tuple_size<InterfaceList>::value;
    std::vector<InterfaceValue> interfaces_map(interfaces_num);
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
};
}  // namespace ir
