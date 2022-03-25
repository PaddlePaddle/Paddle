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

#include <map>  // NOLINT
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/common/scalar.h"

namespace phi {
namespace tests {

TEST(Scalar, ConstructFromDenseTensor) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::GPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<float>(&dense_x);
  dense_x_data[0] = 2.2;
  phi::Scalar scalar_test(dense_x);
  ASSERT_NEAR(dense_x_data[0], scalar_test.to<float>(), 1e-6);
}

TEST(Scalar, ConstructFromTensor) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::GPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<float>(&dense_x);
  dense_x_data[0] = 2.2;

  paddle::experimental::Tensor x(dense_x);
  phi::Scalar scalar_test(x);
  ASSERT_NEAR(dense_x_data[0], scalar_test.to<float>(), 1e-6);
}

}  // namespace tests
}  // namespace phi
