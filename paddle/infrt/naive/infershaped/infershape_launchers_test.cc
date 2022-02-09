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

#include "paddle/infrt/naive/infershaped/infershaped_kernel_launcher.h"
#include "paddle/infrt/naive/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/naive/infershaped/infershaped_registry.h"
#include "paddle/infrt/naive/infershaped/infershaped_utils.h"
#include "paddle/infrt/tensor/dense_host_tensor.h"

namespace infrt {
namespace naive {

namespace {
static void ElementwiseAddTest(const tensor::DenseHostTensor& a,
                               const tensor::DenseHostTensor& b,
                               tensor::DenseHostTensor* c);
}

TEST(utils, registry) {
  constexpr uint8_t count =
      InferShapeHelper<decltype(&ElementwiseAddTest)>::count;
  CHECK_EQ(count, 2U);
}

TEST(ElementwiseAdd, registry) {
  InferShapedKernelRegistry registry;
  RegisterInferShapeLaunchers(&registry);
  ASSERT_EQ(registry.size(), 1UL);
  auto creator = registry.GetKernel("elementwise_add");
  auto infershape_launcher_handle = creator();
  // fake some tensors

  tensor::DenseHostTensor a({2, 8}, GetDType<float>());
  tensor::DenseHostTensor b({2, 8}, GetDType<float>());
  tensor::DenseHostTensor c({2, 8}, GetDType<float>());

  host_context::KernelFrameBuilder kernel_frame_builder;
  kernel_frame_builder.AddArgument(new host_context::Value(0));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(a)));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(b)));
  kernel_frame_builder.SetResults({new host_context::Value(std::move(c))});

  infershape_launcher_handle->Invoke(&kernel_frame_builder);
}

}  // namespace naive
}  // namespace infrt
