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
#include "paddle/pten/api/ext/exception.h"

namespace pten {

struct DeviceContext::Impl {
  Impl() = default;
  ~Impl() = default;

  void SetDeviceAllocator(Allocator* allocator) {
    device_allocator_ = allocator;
  }

  void SetHostAllocator(Allocator* allocator) { host_allocator_ = allocator; }

  const Allocator& GetDeviceAllocator() const {
    PD_CHECK(device_allocator_ != nullptr, "the device_allocator is nullptr.");
    return *device_allocator_;
  }

  const Allocator& GetHostAllocator() const {
    PD_CHECK(host_allocator_ != nullptr, "the host_allocator is nullptr.");
    return *host_allocator_;
  }

  // TODO(Wilber): Add impl. It seems that tensorbase not have interface to
  // communicate with allocator.
  void HostAlloc(TensorBase* tensor) {}
  void DeviceAlloc(TensorBase* tensor) {}

  Allocator* device_allocator_{nullptr};
  Allocator* host_allocator_{nullptr};
};

DeviceContext::DeviceContext() { impl_ = std::make_unique<Impl>(); }

DeviceContext::DeviceContext(const DeviceContext& other) {
  impl_->SetDeviceAllocator(
      const_cast<Allocator*>(&other.GetDeviceAllocator()));
  impl_->SetHostAllocator(const_cast<Allocator*>(&other.GetHostAllocator()));
}

DeviceContext::DeviceContext(DeviceContext&& other) {
  impl_ = std::move(other.impl_);
}

DeviceContext::~DeviceContext() = default;

void DeviceContext::SetHostAllocator(Allocator* allocator) {
  impl_->SetHostAllocator(allocator);
}

void DeviceContext::SetDeviceAllocator(Allocator* allocator) {
  impl_->SetDeviceAllocator(allocator);
}

const Allocator& DeviceContext::GetHostAllocator() const {
  return impl_->GetHostAllocator();
}

const Allocator& DeviceContext::GetDeviceAllocator() const {
  return impl_->GetDeviceAllocator();
}

void DeviceContext::HostAlloc(TensorBase* tensor) { impl_->HostAlloc(tensor); }

void DeviceContext::DeviceAlloc(TensorBase* tensor) {
  impl_->DeviceAlloc(tensor);
}

}  // namespace pten
