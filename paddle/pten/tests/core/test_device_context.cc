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
#include "paddle/pten/backends/cpu/cpu_context.h"
// #include "paddle/pten/backends/all_context.h"

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

namespace pten {
namespace tests {

TEST(DeviceContext, cpu_context) {
  std::cout << "test training scenarios" << std::endl;
  {
    pten::CPUContext ctx;
    CHECK(ctx.eigen_device() != nullptr);
  }

  std::cout << "test inference scenarios" << std::endl;
  Eigen::DefaultDevice* device = new Eigen::DefaultDevice();
  {
    pten::CPUContextResource ctx_res{device};
    pten::CPUContext ctx(ctx_res);
    CHECK(ctx.eigen_device() != nullptr);
  }
  {
    pten::CPUContextResource ctx_res{nullptr};
    pten::CPUContext ctx(ctx_res);
    ctx.SetEigenDevice(device);
    CHECK(ctx.eigen_device() != nullptr);
  }
  delete device;

  std::cout << "test copy constructor" << std::endl;
  {
    pten::CPUContext ctx1;
    pten::CPUContext ctx2(ctx1);
    CHECK_EQ(ctx1.eigen_device(), ctx2.eigen_device());
  }

  std::cout << "test move constructor" << std::endl;
  {
    pten::CPUContext ctx1 = pten::CPUContext();
    auto* eigen_device1 = ctx1.eigen_device();
    pten::CPUContext ctx2(std::move(ctx1));
    auto* eigen_device2 = ctx2.eigen_device();
    CHECK_EQ(eigen_device1, eigen_device2);
  }
}

}  // namespace tests
}  // namespace pten
