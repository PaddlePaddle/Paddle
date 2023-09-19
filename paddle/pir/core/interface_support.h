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

#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/interface_value.h"

namespace pir {
namespace detail {
template <typename ConcreteT, typename... Args>
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
    p_interface->swap(InterfaceValue::get<ConcreteT, T>());
    VLOG(6) << "New a interface: id["
            << (p_interface->type_id()).AsOpaquePointer() << "].";
    ++p_interface;
  }

  /// Placement new trait.
  template <typename T>
  static void PlacementConstrctTrait(pir::TypeId *&p_trait) {  // NOLINT
    *p_trait = TypeId::get<T>();
    VLOG(6) << "New a trait: id[" << p_trait->AsOpaquePointer() << "].";
    ++p_trait;
  }
};

/// Specialized for tuple type.
template <typename ConcreteT, typename... Args>
class ConstructInterfacesOrTraits<ConcreteT, std::tuple<Args...>> {
 public:
  /// Construct method for interfaces.
  static InterfaceValue *interface(InterfaceValue *p_interface) {
    return ConstructInterfacesOrTraits<ConcreteT, Args...>::interface(
        p_interface);
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    return ConstructInterfacesOrTraits<ConcreteT, Args...>::trait(p_trait);
  }
};

template <typename T>
void *LookUp(const TypeId &interface_id,
             const uint32_t num_interfaces,
             const uint32_t num_traits,
             const T *t) {
  if (num_interfaces > 0) {
    const InterfaceValue *p_first_interface =
        reinterpret_cast<const InterfaceValue *>(
            reinterpret_cast<const char *>(t) - sizeof(TypeId) * num_traits -
            sizeof(InterfaceValue) * num_interfaces);
    size_t left = 0, right = num_interfaces;
    while (left < right) {
      size_t mid = (left + right) / 2;
      if ((p_first_interface + mid)->type_id() == interface_id) {
        return (p_first_interface + mid)->model();
      } else if ((p_first_interface + mid)->type_id() < interface_id) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
  }
  return nullptr;
}

template <typename ConcreteT, typename InterfaceList>
std::vector<InterfaceValue> GetInterfaceMap() {
  constexpr size_t interfaces_num = std::tuple_size<InterfaceList>::value;
  std::vector<InterfaceValue> interfaces_map(interfaces_num);
  ConstructInterfacesOrTraits<ConcreteT, InterfaceList>::interface(
      interfaces_map.data());
  return interfaces_map;
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
