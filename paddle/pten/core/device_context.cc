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

namespace pten {

struct DeviceContext::Impl {
  Allocator* allocator_{nullptr};

  Impl() = default;
  ~Impl() = default;

  void SetAllocator(Allocator* allocator) { allocator_ = allocator; }

  const Allocator& GetAllocator() const { return *allocator_; }

  // TODO(Wilber): Add impl. It seems that tensorbase not have interface to
  // communicate with allocator.
  void Alloc(TensorBase* tensor) {}
};

DeviceContext::DeviceContext() { impl_ = std::make_unique<Impl>(); }

DeviceContext::DeviceContext(const DeviceContext& other) {
  impl_->SetAllocator(const_cast<Allocator*>(&other.GetAllocator()));
}

DeviceContext::DeviceContext(DeviceContext&& other) {
  impl_ = std::move(other.impl_);
}

DeviceContext::~DeviceContext() = default;

void DeviceContext::SetAllocator(Allocator* allocator) {
  impl_->SetAllocator(allocator);
}

const Allocator& DeviceContext::GetAllocator() const {
  return impl_->GetAllocator();
}

void DeviceContext::Alloc(TensorBase* tensor) { impl_->Alloc(tensor); }

}  // namespace pten
