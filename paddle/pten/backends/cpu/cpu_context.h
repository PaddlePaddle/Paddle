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

class CPUContext : public DeviceContext {
 public:
  CPUContext();
  virtual ~CPUContext();
  Eigen::DefaultDevice* eigen_device() const;
  const Place& GetPlace() const override;

 protected:
  struct ref_tag {};
  explicit CPUContext(ref_tag);
  void SetEigenDevice(Eigen::DefaultDevice* device);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pten
