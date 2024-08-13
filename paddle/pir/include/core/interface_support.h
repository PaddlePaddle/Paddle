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

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/interface_value.h"

namespace pir {
namespace detail {

template <typename ConcreteT, typename... Args>
class ConstructInterfacesOrTraits {
 public:
  /// Construct method for interfaces.
  static void interface(InterfaceSet &interface_set) {  // NOLINT
    (void)std::initializer_list<int>{
        0, (ConstructInterface<Args>(interface_set), 0)...};
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    (void)std::initializer_list<int>{
        0, (PlacementConstructTrait<Args>(p_trait), 0)...};
    return p_trait;
  }

 private:
  /// Placement new interface.
  template <typename T>
  static void ConstructInterface(InterfaceSet &interface_set) {  // NOLINT
    InterfaceValue val =
        InterfaceValue::Get<T, typename T::template Model<ConcreteT>>();
    auto success = interface_set.insert(std::move(val)).second;
    PADDLE_ENFORCE_EQ(
        success,
        true,
        common::errors::PreconditionNotMet(
            "Interface: id[%u] is already registered. inset failed",
            TypeId::get<T>()));
  }

  /// Placement new trait.
  template <typename T>
  static void PlacementConstructTrait(pir::TypeId *&p_trait) {  // NOLINT
    *p_trait = TypeId::get<T>();
    ++p_trait;
  }
};

/// Specialized for tuple type.
template <typename ConcreteT, typename... Args>
class ConstructInterfacesOrTraits<ConcreteT, std::tuple<Args...>> {  // NOLINT
 public:
  /// Construct method for interfaces.
  static void interface(InterfaceSet &interface_set) {  // NOLINT
    ConstructInterfacesOrTraits<ConcreteT, Args...>::interface(interface_set);
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    return ConstructInterfacesOrTraits<ConcreteT, Args...>::trait(p_trait);
  }
};

template <typename ConcreteT, typename InterfaceList>
InterfaceSet GetInterfaceSet() {
  InterfaceSet interfaces_set;
  ConstructInterfacesOrTraits<ConcreteT, InterfaceList>::interface(
      interfaces_set);
  return interfaces_set;
}

template <typename ConcreteT, typename TraitList>
std::vector<TypeId> GetTraitSet() {
  constexpr size_t traits_num = std::tuple_size<TraitList>::value;
  std::vector<TypeId> trait_set(traits_num);
  auto p_first_trait = trait_set.data();
  ConstructInterfacesOrTraits<ConcreteT, TraitList>::trait(p_first_trait);
  return trait_set;
}

}  // namespace detail

}  // namespace pir
