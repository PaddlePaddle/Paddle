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

#include "paddle/pten/kernels/matmul_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;

TEST(DEV_API, dot) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  DenseTensor dense_x(alloc.get(),
                      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                            framework::make_ddim({3, 3}),
                                            pten::DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();

  DenseTensor dense_y(alloc.get(),
                      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                            framework::make_ddim({3, 3}),
                                            pten::DataLayout::NCHW));
  auto* dense_y_data = dense_y.mutable_data<float>();

  for (size_t i = 0; i < 9; ++i) {
    dense_x_data[i] = 1.0;
    dense_y_data[i] = 2.0;
  }
  std::vector<float> sum(9, 6.0);

  // 2. test API
  pten::CPUContext dev_ctx;
  auto out = Matmul<float, CPUContext>(dev_ctx, dense_x, dense_y, false, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.dtype(), DataType::FLOAT32);
  ASSERT_EQ(out.layout(), DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i], out.data<float>()[i], 1e-6f);
  }
}

}  // namespace tests
}  // namespace pten
