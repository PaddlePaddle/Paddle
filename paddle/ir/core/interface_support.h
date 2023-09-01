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
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/utils.h"

namespace ir {

class IR_API InterfaceValue {
 public:
  // ConcreteOp -> Concrete?
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

// ConcreteOp -> Concrete？
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
    VLOG(6) << "New a interface: id["
            << (p_interface->type_id()).AsOpaquePointer() << "].";
    ++p_interface;
  }

  /// Placement new trait.
  template <typename T>
  static void PlacementConstrctTrait(ir::TypeId *&p_trait) {  // NOLINT
    *p_trait = TypeId::get<T>();
    VLOG(6) << "New a trait: id[" << p_trait->AsOpaquePointer() << "].";
    ++p_trait;
  }
};

// ConcreteOp -> Concrete？
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
}  // namespace ir
