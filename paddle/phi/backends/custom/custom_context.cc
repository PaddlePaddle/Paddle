/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/custom/custom_context.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/stream.h"

namespace phi {

struct CustomContext::Impl {
  explicit Impl(const CustomPlace& place) : place_(place) {}

  ~Impl() {}

  void Init() {
    phi::DeviceGuard guard(place_);
    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
  }

  const Place& GetPlace() const { return place_; }

  void* stream() const {
    return reinterpret_cast<void*>(stream_->raw_stream());
  }

  void Wait() const { stream_->Wait(); }

  Place place_;

  std::shared_ptr<phi::stream::Stream> stream_;
};

void CustomContext::Init() { impl_->Init(); }

const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

void* CustomContext::stream() const { return impl_->stream(); }

void CustomContext::Wait() const { return impl_->Wait(); }

CustomContext::CustomContext(const CustomPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {}

CustomContext::~CustomContext() {}

}  // namespace phi
