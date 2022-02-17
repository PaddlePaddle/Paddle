// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/infrt/kernel/pten/infershaped/infershaped_kernel_launcher.h"
#include "paddle/infrt/kernel/pten/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/pten/infershaped/infershaped_utils.h"
#include "paddle/infrt/kernel/pten/registry.h"

namespace infrt {
namespace kernel {

namespace {
static void ElementwiseAddTest(const ::pten::DenseTensor& a,
                               const ::pten::DenseTensor& b,
                               ::pten::DenseTensor* c);
}

TEST(utils, registry) {
  constexpr uint8_t count =
      InferShapeHelper<decltype(&ElementwiseAddTest)>::count;
  CHECK_EQ(count, 2U);
}

TEST(fake_kernel, launcher_registry) {
  host_context::KernelRegistry registry;
  RegisterInferShapeLaunchers(&registry);
  auto creator = registry.GetKernel("pten_kernel.fake_pten_kernel");
  CHECK(creator);

  ::pten::CPUContext dev_ctx{};
  ::pten::DenseTensor x{};
  ::pten::DenseTensor y{};
  bool transpose_x{};
  ::pten::DenseTensor out{};

  host_context::KernelFrameBuilder kernel_frame_builder;
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(dev_ctx)));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(x)));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(y)));
  kernel_frame_builder.AddArgument(new host_context::Value(transpose_x));
  kernel_frame_builder.SetResults({new host_context::Value(std::move(out))});
  creator(&kernel_frame_builder);
}

}  // namespace kernel
}  // namespace infrt
