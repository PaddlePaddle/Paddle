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
  }

  void destroy() {
    // desctrctor and free memory
  }

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
