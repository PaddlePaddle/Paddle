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

#include "gtest/gtest.h"

// TODO(wilber): will remove after the cpu, gpu context megre.
#include "paddle/phi/backends/cpu/cpu_context.h"
// #include "paddle/phi/backends/all_context.h"

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {
namespace tests {

class InferenceCPUContext : public CPUContext {
 public:
  void SetEigenDevice(Eigen::DefaultDevice* eigen_device) {
    CPUContext::SetEigenDevice(eigen_device);
  }
};

TEST(DeviceContext, cpu_context) {
  std::cout << "test training scenarios" << std::endl;
  {
    phi::CPUContext ctx;
    ctx.Init();
    EXPECT_TRUE(ctx.eigen_device() != nullptr);
  }

  std::cout << "test inference scenarios" << std::endl;
  Eigen::DefaultDevice* device = new Eigen::DefaultDevice();
  {
    InferenceCPUContext ctx;
    ctx.SetEigenDevice(device);
    EXPECT_TRUE(ctx.eigen_device() != nullptr);
  }
  delete device;
}

}  // namespace tests
}  // namespace phi
