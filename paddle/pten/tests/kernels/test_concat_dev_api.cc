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

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/kernels/concat_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;

TEST(DEV_API, concat) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 10}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  pten::DenseTensor dense_y(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 10}),
                            pten::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y.mutable_data<float>(paddle::platform::CPUPlace());

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      dense_y_data[i * 10 + j] = (i * 10 + j) * 1.0;
    }
  }

  std::vector<pten::DenseTensor> inputs = {dense_x, dense_y};

  // 2. test API
  pten::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  dev_ctx.Init();
  auto out = pten::Concat<float>(dev_ctx, inputs, 0);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 6);
  ASSERT_EQ(out.dims()[1], 10);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto out_data = out.data<float>();

  for (size_t i = 0; i < 60; ++i) {
    if (i < 30) {
      ASSERT_NEAR(dense_x_data[i], out_data[i], 1e-6f);
    } else {
      ASSERT_NEAR(dense_y_data[i - 30], out_data[i], 1e-6f);
    }
  }
}

}  // namespace tests
}  // namespace pten
