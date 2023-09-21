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

#include "paddle/pir/core/op_info_impl.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/interface_support.h"

namespace pir {
OpInfo OpInfoImpl::Create(Dialect *dialect,
                          TypeId op_id,
                          const char *op_name,
                          std::vector<InterfaceValue> &&interface_map,
                          const std::vector<TypeId> &trait_set,
                          size_t attributes_num,
                          const char *attributes_name[],  // NOLINT
                          VerifyPtr verify) {
  // (1) Malloc memory for interfaces, traits, opinfo_impl.
  size_t interfaces_num = interface_map.size();
  size_t traits_num = trait_set.size();
  VLOG(6) << "Create OpInfoImpl with: " << interfaces_num << " interfaces, "
          << traits_num << " traits, " << attributes_num << " attributes.";
  size_t base_size = sizeof(InterfaceValue) * interfaces_num +
                     sizeof(TypeId) * traits_num + sizeof(OpInfoImpl);
  char *base_ptr = static_cast<char *>(::operator new(base_size));
  VLOG(6) << "Malloc " << base_size << " Bytes at "
          << static_cast<void *>(base_ptr);
  if (interfaces_num > 0) {
    std::sort(interface_map.begin(), interface_map.end());
    for (size_t index = 0; index < interfaces_num; ++index) {
      new (base_ptr + index * sizeof(InterfaceValue))
          InterfaceValue(std::move(interface_map[index]));
    }
    base_ptr += interfaces_num * sizeof(InterfaceValue);
  }
  if (traits_num > 0) {
    auto p_first_trait = reinterpret_cast<TypeId *>(base_ptr);
    memcpy(base_ptr, trait_set.data(), sizeof(TypeId) * traits_num);
    std::sort(p_first_trait, p_first_trait + traits_num);
    base_ptr += traits_num * sizeof(TypeId);
  }
  // Construct OpInfoImpl.
  VLOG(6) << "Construct OpInfoImpl at " << reinterpret_cast<void *>(base_ptr)
          << " ......";
  OpInfo op_info = OpInfo(new (base_ptr) OpInfoImpl(dialect,
                                                    op_id,
                                                    op_name,
                                                    interfaces_num,
                                                    traits_num,
                                                    attributes_num,
                                                    attributes_name,
                                                    verify));
  return op_info;
}
void OpInfoImpl::Destroy(OpInfo info) {
  if (info.impl_) {
    info.impl_->Destroy();
  } else {
    LOG(WARNING) << "A nullptr OpInfo is destoryed.";
  }
}

pir::IrContext *OpInfoImpl::ir_context() const {
  return dialect_ ? dialect_->ir_context() : nullptr;
}

bool OpInfoImpl::HasTrait(TypeId trait_id) const {
  if (num_traits_ > 0) {
    const TypeId *p_first_trait =
        reinterpret_cast<const TypeId *>(reinterpret_cast<const char *>(this) -
                                         sizeof(pir::TypeId) * num_traits_);
    return std::binary_search(
        p_first_trait, p_first_trait + num_traits_, trait_id);
  }
  return false;
}

bool OpInfoImpl::HasInterface(TypeId interface_id) const {
  if (num_interfaces_ > 0) {
    const InterfaceValue *p_first_interface =
        reinterpret_cast<const InterfaceValue *>(
            reinterpret_cast<const char *>(this) -
            sizeof(pir::TypeId) * num_traits_ -
            sizeof(InterfaceValue) * num_interfaces_);
    return std::binary_search(p_first_interface,
                              p_first_interface + num_interfaces_,
                              InterfaceValue(interface_id));
  }
  return false;
}

void *OpInfoImpl::GetInterfaceImpl(TypeId interface_id) const {
  return pir::detail::LookUp<OpInfoImpl>(
      interface_id, num_interfaces_, num_traits_, this);
}

void OpInfoImpl::Destroy() {
  VLOG(10) << "Destroy op_info impl at " << this;
  // (1) free interfaces
  char *base_ptr = reinterpret_cast<char *>(this) -
                   sizeof(pir::TypeId) * num_traits_ -
                   sizeof(InterfaceValue) * num_interfaces_;
  if (num_interfaces_ > 0) {
    InterfaceValue *p_interface_val =
        reinterpret_cast<InterfaceValue *>(base_ptr);
    for (size_t i = 0; i < num_interfaces_; i++) {
      (p_interface_val + i)->~InterfaceValue();
    }
  }
  // (2) free memeory
  VLOG(10) << "Free base_ptr " << reinterpret_cast<void *>(base_ptr);
  delete base_ptr;
}

}  // namespace pir
