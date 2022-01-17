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

#include "paddle/pten/kernels/complex_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(DEV_API, conj) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc.get(),
                            pten::DenseTensorMeta(pten::DataType::COMPLEX64,
                                                  framework::make_ddim({3, 4}),
                                                  pten::DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<paddle::complex64>();
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = paddle::complex64(i * 1.0, i * 1.0);
  }

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto out = pten::Conj<paddle::complex64>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), dense_x);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.meta().dtype, pten::DataType::COMPLEX64);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto actual_result = out.data<paddle::complex64>();

  for (size_t i = 0; i < 12; ++i) {
    ASSERT_NEAR(i * 1.0, actual_result[i].real, 1e-6f);
    ASSERT_NEAR(i * -1.0, actual_result[i].imag, 1e-6f);
  }
}

}  // namespace tests
}  // namespace pten
