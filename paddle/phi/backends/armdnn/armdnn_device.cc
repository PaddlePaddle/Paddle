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

#include "paddle/phi/backends/armdnn/armdnn_device.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/expect.h"

namespace phi {

struct ArmDNNDevice::Impl {
  Impl() {}

  ~Impl() {
    if (device_) {
      armdnnlibrary::device_close(device_);
    }
  }

  void Init() {
    owned_ = true;
    device_ = armdnnlibrary::device_open();
    PADDLE_ENFORCE_NE(
        device_,
        nullptr,
        phi::errors::Unavailable("The ArmDNNLibrary device is nullptr."));
  }

  void* device() const { return device_; }

  bool owned_{false};
  void* device_{nullptr};
};

ArmDNNDevice::ArmDNNDevice() : impl_(std::make_unique<Impl>()) {
  impl_->Init();
}

ArmDNNDevice::~ArmDNNDevice() = default;

void* ArmDNNDevice::device() const { return impl_->device(); }

ArmDNNDevice& ArmDNNDevice::Singleton() {
  static ArmDNNDevice device;
  return device;
}

}  // namespace phi
