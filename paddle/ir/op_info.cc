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
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/op_info_impl.h"

namespace ir {
bool OpInfo::HasTrait(TypeId trait_id) const {
  return impl_ && impl_->HasTrait(trait_id);
}

bool OpInfo::HasInterface(TypeId interface_id) const {
  return impl_ && impl_->HasInterface(interface_id);
}

IrContext *OpInfo::ir_context() const {
  return impl_ ? impl_->ir_context() : nullptr;
}

const char *OpInfo::name() const { return impl_ ? impl_->name() : nullptr; }

TypeId OpInfo::id() const { return impl_ ? impl_->id() : TypeId(); }

void OpInfo::verify(const std::vector<OpResult> &inputs,
                    const std::vector<Type> &outputs,
                    const AttributeMap &attributes) {
  impl_->verify()(inputs, outputs, attributes);
}

void *OpInfo::GetInterfaceImpl(TypeId interface_id) const {
  return impl_ ? impl_->interface_impl(interface_id) : nullptr;
}

ir::IrContext *OpInfoImpl::ir_context() const {
  return dialect()->ir_context();
}

void *OpInfoImpl::interface_impl(TypeId interface_id) const {
  if (num_interfaces_ > 0) {
    const InterfaceValue *p_first_interface =
        reinterpret_cast<const InterfaceValue *>(
            reinterpret_cast<const char *>(this) -
            sizeof(TypeId) * num_traits_ -
            sizeof(InterfaceValue) * num_interfaces_);
    size_t left = 0, right = num_interfaces_;
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
bool OpInfoImpl::HasTrait(TypeId trait_id) const {
  if (num_traits_ > 0) {
    const TypeId *p_first_trait =
        reinterpret_cast<const TypeId *>(reinterpret_cast<const char *>(this) -
                                         sizeof(ir::TypeId) * num_traits_);
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
            sizeof(ir::TypeId) * num_traits_ -
            sizeof(InterfaceValue) * num_interfaces_);
    return std::binary_search(p_first_interface,
                              p_first_interface + num_interfaces_,
                              InterfaceValue(interface_id));
  }
  return false;
}

OpInfoImpl *OpInfoImpl::create(Dialect *dialect,
                               TypeId op_id,
                               const char *op_name,
                               std::vector<InterfaceValue> &&interface_map,
                               const std::vector<TypeId> &trait_set,
                               size_t attributes_num,
                               const char *attributes_name[],
                               VerifyPtr verify) {
  // (1) Malloc memory for interfaces, traits, opinfo_impl.
  size_t interfaces_num = interface_map.size();
  size_t traits_num = trait_set.size();
  VLOG(4) << "Create OpInfoImpl with: " << interfaces_num << " interfaces, "
          << traits_num << " traits, " << attributes_num << " attributes.";
  size_t base_size = sizeof(InterfaceValue) * interfaces_num +
                     sizeof(TypeId) * traits_num + sizeof(OpInfoImpl);
  char *base_ptr = static_cast<char *>(::operator new(base_size));
  VLOG(4) << "Malloc " << base_size << " Bytes at "
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
  // Construct opinfo_impl.
  OpInfoImpl *p_opinfo_impl = reinterpret_cast<OpInfoImpl *>(base_ptr);
  VLOG(4) << "Construct op_info_impl at " << p_opinfo_impl << " ......";
  OpInfoImpl *op_info = new (p_opinfo_impl) OpInfoImpl(dialect,
                                                       op_id,
                                                       op_name,
                                                       interfaces_num,
                                                       traits_num,
                                                       attributes_num,
                                                       attributes_name,
                                                       verify

  );
  return op_info;
}

void OpInfoImpl::destroy() {
  VLOG(4) << "Destroy op_info impl at " << this;
  // (1) free interfaces
  char *base_ptr = reinterpret_cast<char *>(this) -
                   sizeof(ir::TypeId) * num_traits_ -
                   sizeof(InterfaceValue) * num_interfaces_;
  if (num_interfaces_ > 0) {
    InterfaceValue *p_interface_val =
        reinterpret_cast<InterfaceValue *>(base_ptr);
    for (size_t i = 0; i < num_interfaces_; i++) {
      (p_interface_val + i)->~InterfaceValue();
    }
  }
  // (2) free memeory
  VLOG(4) << "Free base_ptr " << base_ptr;
  free(base_ptr);
}

}  // namespace ir
