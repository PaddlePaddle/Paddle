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

#include "paddle/phi/backends/armdnn/armdnn_context.h"
#include "paddle/phi/backends/armdnn/armdnn_device.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/expect.h"

namespace phi {

struct ArmDNNContext::Impl {
  Impl() {}

  ~Impl() {
    if (context_) {
      armdnnlibrary::context_destroy(context_);
    }
  }

  void Init() {
    owned_ = true;
    context_ =
        armdnnlibrary::context_create(ArmDNNDevice::Singleton().device());
    PADDLE_ENFORCE_NE(
        context_,
        nullptr,
        phi::errors::Unavailable("The ArmDNNLibrary context is nullptr."));
  }

  void* context() const { return context_; }

  bool owned_{false};
  void* context_{nullptr};
};

ArmDNNContext::ArmDNNContext(const Place& place)
    : CPUContext(place), impl_(std::make_unique<Impl>()) {
  impl_->Init();
}

ArmDNNContext::~ArmDNNContext() = default;

void* ArmDNNContext::context() const { return impl_->context(); }

}  // namespace phi
