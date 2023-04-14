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

#include "paddle/ir/op_info.h"
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
    (PlacementConstrctInterface<Args>(p_interface), ...);
  }

  /// Construct method for traits.
  static void trait(TypeId *p_trait) {
    (PlacementConstrctTrait<Args>(p_trait), ...);
  }

 private:
  /// Placement new interface.
  template <typename T>
  void PlacementConstrctInterface(
      std::pair<TypeId, void *> *&p_interface) {  // NOLINT
    // void* ptmp = malloc(sizeof(T::Model<ConcreteOp>));
    // new (ptmp) T::Model<ConcreteOp>();
    // std::pair<TypeId, void*> pair_tmp = make_pair(TypeId::get<T>(), ptmp);
    // memcpy(p_interface, &pair_tmp, sizeof(std::pair<TypeId, void*>));
    p_interface += 1;
  }
  /// Placement new trait.
  template <typename T>
  void PlacementConstrctTrait(TypeId *&p_trait) {  // NOLINT
    // new (p_trait) TypeId(TypeId::get<T>())
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
  /// OpInfoImpl is: interface_n ... interface_0 | trait_n ... trait_0 |
  /// OpInfoImpl | StrAttribute_0 ... StrAttribute_n
  ///
  template <typename ConcreteOp>
  static OpInfoImpl *create() {
    // todo: First malloc memroy by ConcreteOp's InterfaceList and TraitList,
    // then placement new them.
    // (1) 先malloc，得到 p_first_interface_ 和 p_first_trait_
    // (2) 调用 ConstructInterfaces 和 ConstructTraits 构建:
    // ConstructInterfacesOrTraits<ConcreteOp,
    // ConcreteOp::InterfaceList>::interface(p_first_interface_);
    // ConstructInterfacesOrTraits<ConcreteOp,
    // ConcreteOp::TraitList>::trait(p_first_trait_); (3) placement new opinfo
    // (4) placement new attributes name
  }

  void destroy() {
    // desctrctor and free memory
  }

  ///
  /// \brief Search methods for Trait or Interface.
  ///
  bool HasTrait(TypeId trait_id) {
    // Todo: 二分法搜索
  }

  template <typename Trait>
  bool HasTrait() {
    return HasTrait(TypeId::get<Trait>());
  }

  bool HasInterface(TypeId interface_id) {
    // Todo: 二分法搜索
  }

  template <typename Interface>
  bool HasInterface() {
    return HasInterface(TypeId::get<Interface>());
  }

  template <typename Interface>
  Interface::Concept *GetInterfaceImpl() {
    // Todo: 根据 Interface 的 TypeId, 二分查找
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

const OpInfoImpl *OpInfo::impl() const { return impl_; }

bool OpInfo::operator==(OpInfo other) const { return impl_ == other.impl_; }

bool OpInfo::operator!=(OpInfo other) const { return impl_ != other.impl_; }

OpInfo::operator bool() const { return impl_; }

bool OpInfo::operator!() const { return impl_ == nullptr; }

}  // namespace ir
