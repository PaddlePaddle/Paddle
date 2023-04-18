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

#include <cstring>
#include <initializer_list>
#include <utility>

#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/type.h"

namespace ir {
///
/// \brief Tool template classe for construct interfaces or Traits.
///
template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits {
 public:
  /// Construct method for interfaces.
  static void interface(std::pair<TypeId, void *> *p_interface) {
    (void)std::initializer_list<int>{
        0, (PlacementConstrctInterface<Args>(p_interface), 0)...};
  }

  /// Construct method for traits.
  static void trait(TypeId *p_trait) {
    (void)std::initializer_list<int>{
        0, (PlacementConstrctTrait<Args>(p_trait), 0)...};
  }

 private:
  /// Placement new interface.
  template <typename T>
  static void PlacementConstrctInterface(
      std::pair<TypeId, void *> *&p_interface) {  // NOLINT
    void *ptmp = malloc(sizeof(typename T::template Model<ConcreteOp>));
    new (ptmp) typename T::template Model<ConcreteOp>();
    std::pair<ir::TypeId, void *> pair_tmp =
        std::make_pair(ir::TypeId::get<T>(), ptmp);
    memcpy(reinterpret_cast<void *>(p_interface),
           reinterpret_cast<void *>(&pair_tmp),
           sizeof(std::pair<ir::TypeId, void *>));
    p_interface += 1;
  }
  /// Placement new trait.
  template <typename T>
  static void PlacementConstrctTrait(ir::TypeId *&p_trait) {  // NOLINT
    new (p_trait) TypeId(ir::TypeId::get<T>());
    p_trait += 1;
  }
};

/// Specialized for tuple type.
template <typename ConcreteOp, typename... Args>
class ConstructInterfacesOrTraits<ConcreteOp, std::tuple<Args...>> {
 public:
  /// Construct method for interfaces.
  static void interface(std::pair<TypeId, void *> *p_interface) {
    return ConstructInterfacesOrTraits<ConcreteOp, Args...>::interface(
        p_interface);
  }

  /// Construct method for traits.
  static void trait(TypeId *p_trait) {
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
  /// OpInfoImpl is: std::pair<TypeId, void *>... | TypeId... | OpInfoImpl |
  /// StrAttribute_0 ... StrAttribute_n
  ///
  template <typename ConcreteOp>
  static OpInfoImpl *create() {
    // (1) Malloc memory for interfaces, traits, opinfo_impl and StrAttribues.
    size_t interfaces_num =
        std::tuple_size<typename ConcreteOp::InterfaceList>::value;
    size_t traits_num = std::tuple_size<typename ConcreteOp::TraitList>::value;
    size_t attributes_num = sizeof(ConcreteOp::attributes_name_) /
                            sizeof(ConcreteOp::attributes_name_[0]);
    size_t base_size = sizeof(std::pair<ir::TypeId, void *>) * interfaces_num +
                       sizeof(ir::TypeId) * traits_num + sizeof(OpInfoImpl) +
                       sizeof(ir::StrAttribute) * attributes_num;
    void *base_ptr = malloc(base_size);
    std::pair<ir::TypeId, void *> *p_first_interface =
        reinterpret_cast<std::pair<ir::TypeId, void *> *>(base_ptr);
    ir::TypeId *p_first_trait =
        reinterpret_cast<ir::TypeId *>(p_first_interface + interfaces_num);
    OpInfoImpl *p_opinfo_impl =
        reinterpret_cast<OpInfoImpl *>(p_first_trait + traits_num);
    ir::StrAttribute *p_first_attribute =
        reinterpret_cast<ir::StrAttribute *>(p_opinfo_impl + 1);

    // (2) Construct interfaces and sort by TypeId.
    ConstructInterfacesOrTraits<
        ConcreteOp,
        typename ConcreteOp::InterfaceList>::interface(p_first_interface);
    std::sort(p_first_interface, p_first_interface + interfaces_num);

    // (3) Construct traits and sort by TypeId.
    ConstructInterfacesOrTraits<ConcreteOp, typename ConcreteOp::TraitList>::
        trait(p_first_trait);
    std::sort(p_first_trait, p_first_trait + traits_num);

    // (4) Construct opinfo_impl.
    OpInfoImpl *op_info =
        new (p_opinfo_impl) OpInfoImpl(p_first_interface,
                                       p_first_trait,
                                       attributes_num,
                                       ir::TypeId::get<ConcreteOp>(),
                                       ConcreteOp::name());

    // (5) Construct StrAttributes.
    ir::StrAttribute *p_attribute = p_first_attribute;
    ir::IrContext *ctx = ir::IrContext::Instance();
    for (size_t i = 0; i < attributes_num; i++) {
      new (p_attribute) ir::StrAttribute(ir::StrAttribute::get(
          ctx, std::string(ConcreteOp::attributes_name_[i])));
      p_attribute += 1;
    }
    return op_info;
  }

  void destroy() {
    // (1) free interfaces
    size_t interfaces_num = std::distance(
        p_first_interface_,
        reinterpret_cast<std::pair<ir::TypeId, void *> *>(p_first_trait_));
    for (size_t i = 0; i < interfaces_num; i++) {
      free((p_first_interface_ + i)->second);
    }

    // (2) free memeory
    free(reinterpret_cast<void *>(p_first_interface_));
  }

  ///
  /// \brief Search methods for Trait or Interface.
  ///
  bool HasTrait(TypeId trait_id) {
    size_t traits_num =
        std::distance(p_first_trait_, reinterpret_cast<TypeId *>(this));
    if (traits_num > 0) {
      int left = 0;
      int right = traits_num - 1;
      int mid = 0;
      while (left <= right) {
        mid = (left + right) / 2;
        if (*(p_first_trait_ + mid) == trait_id) {
          return true;
        } else if (*(p_first_trait_ + mid) < trait_id) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return false;
  }

  template <typename Trait>
  bool HasTrait() {
    return HasTrait(TypeId::get<Trait>());
  }

  bool HasInterface(TypeId interface_id) {
    size_t interfaces_num = std::distance(
        p_first_interface_,
        reinterpret_cast<std::pair<ir::TypeId, void *> *>(p_first_trait_));
    if (interfaces_num > 0) {
      int left = 0;
      int right = interfaces_num - 1;
      int mid = 0;
      while (left <= right) {
        mid = (left + right) / 2;
        if ((p_first_interface_ + mid)->first == interface_id) {
          return true;
        } else if ((p_first_interface_ + mid)->first < interface_id) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return false;
  }

  template <typename Interface>
  bool HasInterface() {
    return HasInterface(TypeId::get<Interface>());
  }

  template <typename Interface>
  typename Interface::Concept *GetInterfaceImpl() {
    ir::TypeId interface_id = ir::TypeId::get<Interface>();
    size_t interfaces_num = std::distance(
        p_first_interface_,
        reinterpret_cast<std::pair<ir::TypeId, void *> *>(p_first_trait_));
    if (interfaces_num > 0) {
      int left = 0;
      int right = interfaces_num - 1;
      int mid = 0;
      while (left <= right) {
        mid = (left + right) / 2;
        if ((p_first_interface_ + mid)->first == interface_id) {
          return reinterpret_cast<typename Interface::Concept *>(
              (p_first_interface_ + mid)->second);
        } else if ((p_first_interface_ + mid)->first < interface_id) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return false;
  }

 private:
  OpInfoImpl(std::pair<TypeId, void *> *p_first_interface,
             TypeId *p_first_trait,
             uint32_t num_attributes,
             TypeId op_id,
             const char *op_name)
      : p_first_interface_(p_first_interface),
        p_first_trait_(p_first_trait),
        num_attributes_(num_attributes),
        op_id_(op_id),
        op_name_(op_name) {}

  /// Interface will be recorded by std::pair<TypeId, void*>.
  std::pair<TypeId, void *> *p_first_interface_{nullptr};

  /// Trait will be recorded by TypeId.
  TypeId *p_first_trait_{nullptr};

  /// The number of attributes for this Op.
  uint32_t num_attributes_ = 0;

  /// The TypeId of this Op.
  TypeId op_id_;

  /// The name of this Op.
  const char *op_name_;
};

}  // namespace ir
