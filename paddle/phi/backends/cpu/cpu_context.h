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

#include "paddle/phi/backends/cpu/forwards.h"
#include "paddle/phi/core/device_context.h"

// TODO(wilber): Do we need to use place in phi kernel?
#include "paddle/phi/common/place.h"

namespace phi {

class PADDLE_API CPUContext : public DeviceContext,
                              public TypeInfoTraits<DeviceContext, CPUContext> {
 public:
  CPUContext();
  CPUContext(CPUContext&&);
  CPUContext& operator=(CPUContext&&);
  explicit CPUContext(const Place&);
  virtual ~CPUContext();
  Eigen::DefaultDevice* eigen_device() const;
  const Place& GetPlace() const override;

  static const char* name() { return "CPUContext"; }

 protected:
  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetEigenDevice(Eigen::DefaultDevice* device);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
