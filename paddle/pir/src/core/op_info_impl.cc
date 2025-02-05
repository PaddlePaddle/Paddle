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

#include <glog/logging.h>

#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/interface_support.h"
#include "paddle/pir/src/core/op_info_impl.h"

namespace pir {

void OpInfo::AttachInterface(InterfaceValue &&interface_value) {
  PADDLE_ENFORCE_NOT_NULL(impl_,
                          common::errors::InvalidArgument(
                              "Can't attach interface to a nullptr OpInfo"));
  impl_->AttachInterface(std::move(interface_value));
}

void OpInfoImpl::AttachInterface(InterfaceValue &&interface_value) {
  auto type_id = interface_value.type_id();
  auto success = interface_set_.insert(std::move(interface_value)).second;
  PADDLE_ENFORCE_EQ(
      success,
      true,
      common::errors::InvalidArgument(
          "Interface: id[%u] is already registered. inset failed", type_id));
  VLOG(10) << "Attach a interface: id[" << type_id << "]. to " << op_name_;
}

OpInfoImpl::OpInfoImpl(std::set<InterfaceValue> &&interface_set,
                       pir::Dialect *dialect,
                       TypeId op_id,
                       const char *op_name,
                       uint32_t num_traits,
                       uint32_t num_attributes,
                       const char **p_attributes,
                       VerifyPtr verify_sig,
                       VerifyPtr verify_region)
    : interface_set_(std::move(interface_set)),
      dialect_(dialect),
      op_id_(op_id),
      op_name_(op_name),
      num_traits_(num_traits),
      num_attributes_(num_attributes),
      p_attributes_(p_attributes),
      verify_sig_(verify_sig),
      verify_region_(verify_region) {}

OpInfo OpInfoImpl::Create(Dialect *dialect,
                          TypeId op_id,
                          const char *op_name,
                          std::set<InterfaceValue> &&interface_set,
                          const std::vector<TypeId> &trait_set,
                          size_t attributes_num,
                          const char *attributes_name[],  // NOLINT
                          VerifyPtr verify_sig,
                          VerifyPtr verify_region) {
  // (1) Malloc memory for traits, opinfo_impl.
  size_t traits_num = trait_set.size();
  VLOG(10) << "Create OpInfoImpl with: " << interface_set.size()
           << " interfaces, " << traits_num << " traits, " << attributes_num
           << " attributes.";
  size_t base_size = sizeof(TypeId) * traits_num + sizeof(OpInfoImpl);
  std::unique_ptr<char[]> base_ptr(new char[base_size]);
  VLOG(10) << "Malloc " << base_size << " Bytes at "
           << static_cast<void *>(base_ptr.get());

  char *raw_base_ptr = base_ptr.get();
  if (traits_num > 0) {
    auto p_first_trait = reinterpret_cast<TypeId *>(raw_base_ptr);
    memcpy(raw_base_ptr, trait_set.data(), sizeof(TypeId) * traits_num);
    std::sort(p_first_trait, p_first_trait + traits_num);
    raw_base_ptr += traits_num * sizeof(TypeId);
  }

  // Construct OpInfoImpl.
  VLOG(10) << "Construct OpInfoImpl at "
           << reinterpret_cast<void *>(raw_base_ptr) << " ......";
  OpInfoImpl *impl = new (raw_base_ptr) OpInfoImpl(std::move(interface_set),
                                                   dialect,
                                                   op_id,
                                                   op_name,
                                                   traits_num,
                                                   attributes_num,
                                                   attributes_name,
                                                   verify_sig,
                                                   verify_region);

  // Release the unique_ptr ownership after successful construction
  base_ptr.release();
  return OpInfo(impl);
}
void OpInfoImpl::Destroy(OpInfo info) {
  if (info.impl_) {
    info.impl_->Destroy();
  } else {
    LOG(WARNING) << "A nullptr OpInfo is destroyed.";
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
  return interface_set_.find(interface_id) != interface_set_.end();
}

void *OpInfoImpl::GetInterfaceImpl(TypeId interface_id) const {
  auto iter = interface_set_.find(interface_id);
  return iter != interface_set_.end() ? iter->model() : nullptr;
}

void OpInfoImpl::Destroy() {
  VLOG(10) << "Destroy op_info impl at " << this;
  // (1) compute memory address
  char *base_ptr =
      reinterpret_cast<char *>(this) - sizeof(pir::TypeId) * num_traits_;
  // (2)free interfaces
  this->~OpInfoImpl();
  // (3) free memory
  VLOG(10) << "Free base_ptr " << reinterpret_cast<void *>(base_ptr);
  ::operator delete(base_ptr);
}

std::vector<std::string> OpInfoImpl::GetAttributesName() const {
  std::vector<std::string> attributes_name;
  for (size_t i = 0; i < num_attributes_; ++i) {
    attributes_name.push_back(p_attributes_[i]);
  }
  return attributes_name;
}

}  // namespace pir
