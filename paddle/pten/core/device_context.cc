//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/core/device_context.h"
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/tensor_base.h"

namespace pten {
using DataType = paddle::experimental::DataType;

void DeviceContextImpl::SetDeviceAllocator(const Allocator* allocator) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      pten::errors::InvalidArgument(
          "Required allocator shall not be nullptr, but received nullptr."));
  device_allocator_ = allocator;
}

void DeviceContextImpl::SetHostAllocator(const Allocator* allocator) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      pten::errors::InvalidArgument(
          "Required allocator shall not be nullptr, but received nullptr."));
  host_allocator_ = allocator;
}

void DeviceContextImpl::SetZeroAllocator(const Allocator* allocator) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      pten::errors::InvalidArgument(
          "Required allocator shall not be nullptr, but received nullptr."));
  zero_allocator_ = allocator;
}

const Allocator* DeviceContextImpl::GetDeviceAllocator() const {
  return device_allocator_;
}

const Allocator* DeviceContextImpl::GetHostAllocator() const {
  return host_allocator_;
}

const Allocator* DeviceContextImpl::GetZeroAllocator() const {
  return zero_allocator_;
}

const Allocator* DeviceContext::GetZeroAllocator() const {
  return impl_->GetZeroAllocator();
}

void* DeviceContextImpl::Alloc(TensorBase* tensor,
                               DataType dtype,
                               size_t requested_size) const {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      pten::errors::InvalidArgument(
          "Required tensor shall not be nullptr, but received nullptr."));
  if (dtype == DataType::UNDEFINED) {
    dtype = tensor->dtype();
  }
  auto* allocator = tensor->numel() == 0 ? zero_allocator_ : device_allocator_;
  return tensor->AllocateFrom(
      const_cast<Allocator*>(allocator), dtype, requested_size);
}

void* DeviceContextImpl::HostAlloc(TensorBase* tensor,
                                   DataType dtype,
                                   size_t requested_size) const {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      pten::errors::InvalidArgument(
          "Required tensor shall not be nullptr, but received nullptr."));
  if (dtype == DataType::UNDEFINED) {
    dtype = tensor->dtype();
  }
  auto* allocator = tensor->numel() == 0 ? zero_allocator_ : device_allocator_;
  return tensor->AllocateFrom(
      const_cast<Allocator*>(allocator), dtype, requested_size);
}

DeviceContext::DeviceContext() {
  impl_ = std::make_unique<DeviceContextImpl>();
}

DeviceContext::DeviceContext(const DeviceContext& other) {
  impl_->SetHostAllocator(other.GetHostAllocator());
  impl_->SetDeviceAllocator(other.GetDeviceAllocator());
  impl_->SetZeroAllocator(other.GetZeroAllocator());
}

DeviceContext::DeviceContext(DeviceContext&& other) {
  impl_ = std::move(other.impl_);
}

DeviceContext::~DeviceContext() = default;

void DeviceContext::SetDeviceAllocator(const Allocator* allocator) {
  impl_->SetDeviceAllocator(allocator);
}

const Allocator* DeviceContext::GetDeviceAllocator() const {
  return impl_->GetDeviceAllocator();
}

void DeviceContext::SetHostAllocator(const Allocator* allocator) {
  impl_->SetHostAllocator(allocator);
}

void DeviceContext::SetZeroAllocator(const Allocator* allocator) {
  impl_->SetZeroAllocator(allocator);
}

const Allocator* DeviceContext::GetHostAllocator() const {
  return impl_->GetHostAllocator();
}

void* DeviceContext::Alloc(TensorBase* tensor,
                           DataType dtype,
                           size_t requested_size) const {
  return impl_->Alloc(tensor, dtype, requested_size);
}

void* DeviceContext::HostAlloc(TensorBase* tensor,
                               DataType dtype,
                               size_t requested_size) const {
  return impl_->HostAlloc(tensor, dtype, requested_size);
}

}  // namespace pten
