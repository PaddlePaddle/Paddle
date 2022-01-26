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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/kernels/empty_kernel.h"
#include "paddle/pten/kernels/full_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;

TEST(DEV_API, empty) {
  // 1. create input
  pten::CPUContext dev_ctx;

  // 2. test API
  auto out = pten::Empty<float>(dev_ctx, {3, 2}, pten::DataType::INT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.meta().dtype, pten::DataType::INT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);
}

TEST(DEV_API, empty_like) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());
  dense_x_data[0] = 0;

  // 2. test API
  pten::CPUContext dev_ctx;
  auto out = pten::EmptyLike<float>(dev_ctx, dense_x);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);
}

TEST(DEV_API, full) {
  // 1. create input
  float val = 1.0;

  // 2. test API
  pten::CPUContext dev_ctx;
  auto out = pten::Full<float>(dev_ctx, {3, 2}, val, pten::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto* actual_result = out.data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(DEV_API, full_like) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());
  dense_x_data[0] = 0;
  float val = 1.0;

  pten::CPUContext dev_ctx;

  // 2. test API
  auto out = pten::FullLike<float>(dev_ctx, dense_x, val);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto* actual_result = out.data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

}  // namespace tests
}  // namespace pten
