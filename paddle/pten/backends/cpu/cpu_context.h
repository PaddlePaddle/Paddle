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

#pragma once

#include <memory>

#include "paddle/pten/backends/cpu/forwards.h"
#include "paddle/pten/core/device_context.h"

// TODO(wilber): Do we need to use place in pten kernel?
#include "paddle/pten/common/place.h"

namespace pten {

struct CPUContextResource {
  Eigen::DefaultDevice* device{nullptr};
};

class CPUContext : public DeviceContext {
 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  CPUContext();

  // NOTE: Share the same underlying resources, please ensure that resources are
  // not released.
  CPUContext(const CPUContext&);

  CPUContext(CPUContext&&);

  ~CPUContext();

  Eigen::DefaultDevice* eigen_device() const;

  // TODO(wilber): Whether the interface should be preserved.
  Place GetPlace() const override;

 public:
  // NOTE: External users manage resources. Used in inference scenarios.
  explicit CPUContext(const CPUContextResource& ctx_res);

  void SetEigenDevice(Eigen::DefaultDevice* device);

 private:
  struct CPUImpl;
  std::unique_ptr<CPUImpl> cpu_impl_;
};

}  // namespace pten
