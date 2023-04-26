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

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "paddle/ir/builtin_attribute.h"
// #include "paddle/ir/ir_context.h"
#include "paddle/ir/type.h"

namespace ir {
class Dialect;
///
/// \brief Tool template class for construct interfaces or Traits.
///
template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits {
 public:
  /// Construct method for interfaces.
  static std::pair<TypeId, void *> *interface(
      std::pair<TypeId, void *> *p_interface) {
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
      std::pair<TypeId, void *> *&p_interface) {  // NOLINT
    new (&(p_interface->first)) TypeId(ir::TypeId::get<T>());
    p_interface->second =
        malloc(sizeof(typename T::template Model<ConcreteOp>));
    new (p_interface->second) typename T::template Model<ConcreteOp>();
    VLOG(4) << "New a interface: id[" << p_interface->first.storage()
            << "], interface[" << p_interface->second << "].";
    ++p_interface;
  }

  /// Placement new trait.
  template <typename T>
  static void PlacementConstrctTrait(ir::TypeId *&p_trait) {  // NOLINT
    new (p_trait) TypeId(ir::TypeId::get<T>());
    VLOG(4) << "New a trait: id[" << (*p_trait).storage() << "].";
    ++p_trait;
  }
};

/// Specialized for tuple type.
template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits<ConcreteOp, std::tuple<Args...>> {
 public:
  /// Construct method for interfaces.
  static std::pair<TypeId, void *> *interface(
      std::pair<TypeId, void *> *p_interface) {
    return ConstructInterfacesOrTraits<ConcreteOp, Args...>::interface(
        p_interface);
  }

  /// Construct method for traits.
  static TypeId *trait(TypeId *p_trait) {
    return ConstructInterfacesOrTraits<ConcreteOp, Args...>::trait(p_trait);
  }
};

///
/// \brief OpInfoImpl class.
///
class OpInfoImpl {
 public:
  ///
  /// \brief Construct and Deconstruct OpInfoImpl. The memory layout of
  /// OpInfoImpl is: std::pair<TypeId, void *>... | TypeId... | OpInfoImpl
  ///
  template <typename ConcreteOp>
  static OpInfoImpl *create(ir::Dialect *dialect) {
    // (1) Malloc memory for interfaces, traits, opinfo_impl.
    size_t interfaces_num =
        std::tuple_size<typename ConcreteOp::InterfaceList>::value;
    size_t traits_num = std::tuple_size<typename ConcreteOp::TraitList>::value;
    size_t attributes_num = ConcreteOp::attributes_num();
    VLOG(4) << "Create OpInfoImpl with: " << interfaces_num << " interfaces, "
            << traits_num << " traits, " << attributes_num << " attributes.";
    size_t base_size = sizeof(std::pair<ir::TypeId, void *>) * interfaces_num +
                       sizeof(ir::TypeId) * traits_num + sizeof(OpInfoImpl);
    void *base_ptr = malloc(base_size);
    VLOG(4) << "Malloc " << base_size << " Bytes at " << base_ptr;

    // (2) Construct interfaces and sort by TypeId.
    std::pair<ir::TypeId, void *> *p_first_interface = nullptr;
    if (interfaces_num > 0) {
      p_first_interface =
          reinterpret_cast<std::pair<ir::TypeId, void *> *>(base_ptr);
      VLOG(4) << "Construct interfaces at " << p_first_interface << " ......";
      ConstructInterfacesOrTraits<
          ConcreteOp,
          typename ConcreteOp::InterfaceList>::interface(p_first_interface);
      std::sort(p_first_interface, p_first_interface + interfaces_num);
      base_ptr = reinterpret_cast<void *>(p_first_interface + interfaces_num);
    }

    // (3) Construct traits and sort by TypeId.
    ir::TypeId *p_first_trait = nullptr;
    if (traits_num > 0) {
      p_first_trait = reinterpret_cast<ir::TypeId *>(base_ptr);
      VLOG(4) << "Construct traits at " << p_first_trait << " ......";
      ConstructInterfacesOrTraits<ConcreteOp, typename ConcreteOp::TraitList>::
          trait(p_first_trait);
      std::sort(p_first_trait, p_first_trait + traits_num);
      base_ptr = reinterpret_cast<void *>(p_first_trait + traits_num);
    }

    // (4) Construct opinfo_impl.
    OpInfoImpl *p_opinfo_impl = reinterpret_cast<OpInfoImpl *>(base_ptr);
    VLOG(4) << "Construct op_info_impl at " << p_opinfo_impl << " ......";
    OpInfoImpl *op_info =
        new (p_opinfo_impl) OpInfoImpl(interfaces_num,
                                       traits_num,
                                       ConcreteOp::attributes_name_,
                                       attributes_num,
                                       ir::TypeId::get<ConcreteOp>(),
                                       ConcreteOp::name(),
                                       dialect);
    return op_info;
  }

  void destroy() {
    VLOG(4) << "Destroy op_info impl at " << this;
    // (1) free interfaces
    void *base_ptr = reinterpret_cast<void *>(
        reinterpret_cast<char *>(this) - sizeof(ir::TypeId) * num_traits_ -
        sizeof(std::pair<ir::TypeId, void *>) * num_interfaces_);
    if (num_interfaces_ > 0) {
      std::pair<ir::TypeId, void *> *p_first_interface =
          reinterpret_cast<std::pair<ir::TypeId, void *> *>(base_ptr);
      for (size_t i = 0; i < num_interfaces_; i++) {
        free((p_first_interface + i)->second);
      }
    }
    // (2) free memeory
    VLOG(4) << "Free base_ptr " << base_ptr;
    free(base_ptr);
  }

  ///
  /// \brief Search methods for Trait or Interface.
  ///
  template <typename Trait>
  bool HasTrait() const {
    return HasTrait(TypeId::get<Trait>());
  }

  bool HasTrait(TypeId trait_id) const {
    if (num_traits_ > 0) {
      TypeId *p_first_trait = reinterpret_cast<TypeId *>(
          reinterpret_cast<char *>(const_cast<OpInfoImpl *>(this)) -
          sizeof(ir::TypeId) * num_traits_);
      return std::binary_search(
          p_first_trait, p_first_trait + num_traits_, trait_id);
    }
    return false;
  }

  template <typename Interface>
  bool HasInterface() const {
    return HasInterface(TypeId::get<Interface>());
  }

  bool HasInterface(TypeId interface_id) const {
    if (num_interfaces_ > 0) {
      std::pair<ir::TypeId, void *> *p_first_interface =
          reinterpret_cast<std::pair<ir::TypeId, void *> *>(
              reinterpret_cast<char *>(const_cast<OpInfoImpl *>(this)) -
              sizeof(ir::TypeId) * num_traits_ -
              sizeof(std::pair<ir::TypeId, void *>) * num_interfaces_);
      return std::binary_search(p_first_interface,
                                p_first_interface + num_interfaces_,
                                std::make_pair(interface_id, nullptr),
                                CompareInterface);
    }
    return false;
  }

  template <typename Interface>
  typename Interface::Concept *GetInterfaceImpl() const {
    if (num_interfaces_ > 0) {
      ir::TypeId interface_id = ir::TypeId::get<Interface>();
      std::pair<ir::TypeId, void *> *p_first_interface =
          reinterpret_cast<std::pair<ir::TypeId, void *> *>(
              reinterpret_cast<char *>(const_cast<OpInfoImpl *>(this)) -
              sizeof(ir::TypeId) * num_traits_ -
              sizeof(std::pair<ir::TypeId, void *>) * num_interfaces_);
      size_t left = 0;
      size_t right = num_interfaces_;
      while (left < right) {
        size_t mid = left + (right - left) / 2;
        if ((p_first_interface + mid)->first == interface_id) {
          return reinterpret_cast<typename Interface::Concept *>(
              (p_first_interface + mid)->second);
        } else if ((p_first_interface + mid)->first < interface_id) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
    }
    return nullptr;
  }

  ir::TypeId id() const { return op_id_; }

  const char *name() const { return op_name_; }

  ir::Dialect *dialect() const { return dialect_; }

 private:
  OpInfoImpl(uint32_t num_interfaces,
             uint32_t num_traits,
             const char **p_attributes,
             uint32_t num_attributes,
             TypeId op_id,
             const char *op_name,
             ir::Dialect *dialect)
      : num_interfaces_(num_interfaces),
        num_traits_(num_traits),
        p_attributes_(p_attributes),
        num_attributes_(num_attributes),
        op_id_(op_id),
        op_name_(op_name),
        dialect_(dialect) {}

  static bool CompareInterface(const std::pair<ir::TypeId, void *> &a,
                               const std::pair<ir::TypeId, void *> &b) {
    return a.first < b.first;
  }

  /// Interface will be recorded by std::pair<TypeId, void*>.
  uint32_t num_interfaces_ = 0;

  /// Trait will be recorded by TypeId.
  uint32_t num_traits_ = 0;

  /// Attributes array address.
  const char **p_attributes_{nullptr};

  /// The number of attributes for this Op.
  uint32_t num_attributes_ = 0;

  /// The TypeId of this Op.
  TypeId op_id_;

  /// The name of this Op.
  const char *op_name_;

  /// The dialect of this Op belong to.
  ir::Dialect *dialect_;
};

}  // namespace ir
