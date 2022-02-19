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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/dot_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

TEST(DEV_API, dot) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  phi::DenseTensor dense_x(alloc.get(),
                           phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                                phi::make_ddim({3, 10}),
                                                phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  phi::DenseTensor dense_y(alloc.get(),
                           phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                                phi::make_ddim({3, 10}),
                                                phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y.mutable_data<float>(paddle::platform::CPUPlace());

  float sum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      dense_y_data[i * 10 + j] = (i * 10 + j) * 1.0;
      sum[i] += (i * 10 + j) * (i * 10 + j) * 1.0;
    }
  }

  // 2. test API
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  dev_ctx.Init();
  auto out = phi::Dot<float>(dev_ctx, dense_x, dense_y);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.meta().dtype, phi::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, phi::DataLayout::NCHW);

  auto expect_result = sum;
  auto actual_result0 = out.data<float>()[0];
  auto actual_result1 = out.data<float>()[1];
  auto actual_result2 = out.data<float>()[2];
  ASSERT_NEAR(expect_result[0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[2], actual_result2, 1e-6f);
}

}  // namespace tests
}  // namespace phi
