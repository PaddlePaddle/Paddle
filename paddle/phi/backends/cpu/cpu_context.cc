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

#include "paddle/phi/backends/cpu/cpu_context.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "paddle/phi/core/device_context.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {

struct CPUContext::Impl {
  Impl() : place_(CPUPlace()) {}

  explicit Impl(const Place& place) : place_(place) {}

  ~Impl() {
    if (owned_) {
      delete eigen_device_;
    }
  }

  void Init() {
    owned_ = true;
    eigen_device_ = new Eigen::DefaultDevice();
  }

  Eigen::DefaultDevice* GetEigenDevice() const {
    PADDLE_ENFORCE_NE(
        eigen_device_,
        nullptr,
        common::errors::Unavailable("the cpu eigen_device is nullptr."));
    return eigen_device_;
  }

  bool owned_{false};
  Eigen::DefaultDevice* eigen_device_{nullptr};
  Place place_;
};

CPUContext::CPUContext()
    : DeviceContext(), impl_(std::make_unique<CPUContext::Impl>()) {
  impl_->Init();
}

CPUContext::CPUContext(const Place& place)
    : DeviceContext(), impl_(std::make_unique<CPUContext::Impl>(place)) {
  impl_->Init();
}

CPUContext::~CPUContext() = default;

CPUContext::CPUContext(CPUContext&&) = default;  // NOLINT

CPUContext& CPUContext::operator=(CPUContext&&) = default;  // NOLINT

Eigen::DefaultDevice* CPUContext::eigen_device() const {
  return impl_->GetEigenDevice();
}

const Place& CPUContext::GetPlace() const { return impl_->place_; }

void CPUContext::SetEigenDevice(Eigen::DefaultDevice* device) {
  impl_->eigen_device_ = device;
}

}  // namespace phi
