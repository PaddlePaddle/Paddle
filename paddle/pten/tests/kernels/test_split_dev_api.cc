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

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/kernels/split_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/pten/api/include/manual_api.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;

TEST(DEV_API, split) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      pten::CPUPlace());
  pten::DenseTensor dense_x(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({4, 10}),
                            pten::DataLayout::NCHW));
  pten::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<float>(&dense_x);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
    }
  }

  // 2. test API
  auto out = pten::Split<float>(dev_ctx, dense_x, {2, 2}, 0);

  // 3. check result
  ASSERT_EQ(out.size(), static_cast<size_t>(2));
  ASSERT_EQ(out[0].dims().size(), 2);
  ASSERT_EQ(out[0].dims()[0], 2);
  ASSERT_EQ(out[0].dims()[1], 10);
  ASSERT_EQ(out[0].meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out[0].meta().layout, pten::DataLayout::NCHW);

  ASSERT_EQ(out[1].dims().size(), 2);
  ASSERT_EQ(out[1].dims()[0], 2);
  ASSERT_EQ(out[1].dims()[1], 10);
  ASSERT_EQ(out[1].meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out[1].meta().layout, pten::DataLayout::NCHW);

  auto out_data_0 = out[0].data<float>();
  auto out_data_1 = out[1].data<float>();
  for (size_t i = 0; i < 4; ++i) {
    if (i < 20) {
      ASSERT_NEAR(dense_x_data[i], out_data_0[i], 1e-6);
    } else {
      ASSERT_NEAR(dense_x_data[i], out_data_1[i - 20], 1e-6);
    }
  }
}

}  // namespace tests
}  // namespace pten
