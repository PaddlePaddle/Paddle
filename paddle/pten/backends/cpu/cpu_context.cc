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

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "paddle/pten/core/device_context.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace pten {

struct CPUContext::CPUImpl {
  Eigen::DefaultDevice* device_{nullptr};
  CPUContextResource res_;
  CPUPlace place_;

  CPUImpl() { device_ = new Eigen::DefaultDevice(); }

  // Users need to manage external resources.
  explicit CPUImpl(const CPUContextResource& ctx_res) : res_(ctx_res) {
    device_ = res_.device;
  }

  ~CPUImpl() {
    if (res_.device == nullptr) {
      delete device_;
      device_ = nullptr;
    }
  }

  Eigen::DefaultDevice* GetEigenDevice() const {
    PD_CHECK(device_ != nullptr, "the eigen_device is nullptr.");
    return device_;
  }

  void SetEigenDevice(Eigen::DefaultDevice* device) {
    if (device == nullptr) {
      return;
    }
    res_.device = device;
    device_ = device;
  }

  Place GetPlace() const { return place_; }
};

CPUContext::CPUContext() : DeviceContext(), cpu_impl_(nullptr) {
  cpu_impl_ = std::make_unique<CPUImpl>();
}

CPUContext::CPUContext(const CPUContext& other)
    : DeviceContext(), cpu_impl_(nullptr) {
  cpu_impl_ = std::make_unique<CPUImpl>();
  cpu_impl_->SetEigenDevice(other.eigen_device());
}

CPUContext::CPUContext(CPUContext&& other)
    : DeviceContext(), cpu_impl_(nullptr) {
  cpu_impl_ = std::move(other.cpu_impl_);
}

CPUContext::~CPUContext() = default;

CPUContext::CPUContext(const CPUContextResource& ctx_res)
    : DeviceContext(), cpu_impl_(nullptr) {
  cpu_impl_ = std::make_unique<CPUImpl>(ctx_res);
}

Eigen::DefaultDevice* CPUContext::eigen_device() const {
  return cpu_impl_->GetEigenDevice();
}

void CPUContext::SetEigenDevice(Eigen::DefaultDevice* device) {
  cpu_impl_->SetEigenDevice(device);
}

Place CPUContext::GetPlace() const { return cpu_impl_->GetPlace(); }

}  // namespace pten
