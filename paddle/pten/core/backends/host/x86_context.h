/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

// TODO(wilber): Remove after adding WITH_EIGEN macro
#define PADDLE_WITH_EIGEN 1
#include "paddle/fluid/platform/place.h"

#include "paddle/pten/core/device_context.h"

#ifdef PADDLE_WITH_EIGEN
#include "unsupported/Eigen/CXX11/Tensor"
namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen
#endif

namespace pten {

using Place = paddle::platform::Place;
using CPUPlace = paddle::platform::CPUPlace;

class CPUContext : public DeviceContext {
 public:
  explicit CPUContext(CPUPlace place) : place_(place) {
#ifdef PADDLE_WITH_EIGEN
    eigen_device_.reset(new Eigen::DefaultDevice());
#endif
  }

  Place GetPlace() const noexcept override { return place_; }

#ifdef PADDLE_WITH_EIGEN
  Eigen::DefaultDevice* eigen_device() const { return eigen_device_.get(); }
#endif

 private:
  CPUPlace place_;

#ifdef PADDLE_WITH_EIGEN
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
#endif
};

}  // namespace pten
