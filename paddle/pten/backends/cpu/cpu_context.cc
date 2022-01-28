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

#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/common/place.h"

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "paddle/pten/core/device_context.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace pten {

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
    PD_CHECK(eigen_device_ != nullptr, "the cpu eigen_device is nullptr.");
    return eigen_device_;
  }

  bool owned_{false};
  Eigen::DefaultDevice* eigen_device_{nullptr};
  Place place_;
};

CPUContext::CPUContext()
    : DeviceContext(), impl_(std::make_unique<CPUContext::Impl>()) {}

CPUContext::CPUContext(const Place& place)
    : DeviceContext(), impl_(std::make_unique<CPUContext::Impl>(place)) {}

CPUContext::~CPUContext() = default;

void CPUContext::Init() { impl_->Init(); }

Eigen::DefaultDevice* CPUContext::eigen_device() const {
  return impl_->GetEigenDevice();
}

const Place& CPUContext::GetPlace() const { return impl_->place_; }

void CPUContext::SetEigenDevice(Eigen::DefaultDevice* device) {
  impl_->eigen_device_ = device;
}

}  // namespace pten
