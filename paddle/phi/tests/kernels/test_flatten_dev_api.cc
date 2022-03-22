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
#include "paddle/phi/kernels/flatten_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(copy, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);
#endif

#ifdef PADDLE_WITH_XPU
PD_DECLARE_KERNEL(copy, XPU, ALL_LAYOUT);
#endif

namespace phi {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

TEST(DEV_API, flatten) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  phi::DenseTensor dense_x(alloc.get(),
                           phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                                phi::make_ddim({3, 2, 2, 3}),
                                                phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  for (int i = 0; i < dense_x.numel(); i++) {
    dense_x_data[i] = i;
  }
  int start_axis = 1, stop_axis = 2;
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  dev_ctx.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx.Init();

  // 2. test API
  auto out = phi::Flatten<float>(dev_ctx, dense_x, start_axis, stop_axis);

  // 3. check result
  std::vector<int> expect_shape = {3, 4, 3};
  ASSERT_EQ(out.dims()[0], expect_shape[0]);
  ASSERT_EQ(out.dims()[1], expect_shape[1]);
  ASSERT_EQ(out.dims()[2], expect_shape[2]);
  ASSERT_EQ(out.numel(), 36);
  ASSERT_EQ(out.meta().dtype, phi::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, phi::DataLayout::NCHW);

  bool value_equal = true;
  auto* dense_out_data = out.data<float>();
  for (int i = 0; i < dense_x.numel(); i++) {
    if (std::abs(dense_x_data[i] - dense_out_data[i]) > 1e-6f)
      value_equal = false;
  }
  ASSERT_EQ(value_equal, true);
}

}  // namespace tests
}  // namespace phi
