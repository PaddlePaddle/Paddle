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

#include "paddle/pten/backends/cpu/cpu_context.h"

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

namespace pten {

struct CPUContext::Impl {
  Eigen::DefaultDevice* device_{nullptr};
  CPUContextResource res_;
  paddle::platform::CPUPlace place_;

  Impl() { device_ = new Eigen::DefaultDevice(); }

  // Users need to manage external resources.
  explicit Impl(const CPUContextResource& ctx_res) : res_(ctx_res) {
    device_ = res_.device;
  }

  ~Impl() {
    if (res_.device == nullptr) {
      delete device_;
    }
  }

  Eigen::DefaultDevice* GetEigenDevice() const { return device_; }

  void SetEigenDevice(Eigen::DefaultDevice* device) {
    if (device == nullptr) {
      return;
    }
    res_.device = device;
    device_ = device;
  }

  paddle::platform::Place GetPlace() const { return place_; }
};

CPUContext::CPUContext() : DeviceContext() { impl_ = std::make_unique<Impl>(); }

CPUContext::CPUContext(const CPUContext& other) : DeviceContext() {
  impl_ = std::make_unique<Impl>();
  impl_->SetEigenDevice(other.eigen_device());
}

CPUContext::~CPUContext() = default;

CPUContext::CPUContext(const CPUContextResource& ctx_res) : DeviceContext() {
  impl_ = std::make_unique<Impl>(ctx_res);
}

Eigen::DefaultDevice* CPUContext::eigen_device() const {
  return impl_->GetEigenDevice();
}

void CPUContext::SetEigenDevice(Eigen::DefaultDevice* device) {
  impl_->SetEigenDevice(device);
}

paddle::platform::Place CPUContext::GetPlace() const {
  return impl_->GetPlace();
}

}  // namespace pten
