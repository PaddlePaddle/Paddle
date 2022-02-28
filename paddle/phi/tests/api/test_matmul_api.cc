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

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

TEST(API, matmul_cpu) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));

  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<float>(paddle::platform::CPUPlace());

  for (size_t i = 0; i < 9; ++i) {
    dense_x_data[i] = 1.0;
    dense_y_data[i] = 2.0;
  }
  std::vector<float> sum(9, 6.0);

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::matmul(x, y, false, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i], dense_out->data<float>()[i], 1e-6f);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(API, matmul_cuda) {
  // Prepare CPU Dense Tensor
  const auto alloc_cpu =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  auto ref_x = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));

  auto* ref_x_data = ref_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto ref_y = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));
  auto* ref_y_data = ref_y->mutable_data<float>(paddle::platform::CPUPlace());

  for (size_t i = 0; i < 9; ++i) {
    ref_x_data[i] = 1.0;
    ref_y_data[i] = 2.0;
  }
  std::vector<float> sum(9, 6.0);

  // 1. create tensor
  const auto alloc_cuda =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc_cuda.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc_cuda.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 3}),
                           phi::DataLayout::NCHW));

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CUDAPlace();
  auto* dev_ctx = static_cast<const phi::GPUContext*>(pool.GetByPlace(place));

  phi::Copy(*dev_ctx, *ref_x.get(), phi::GPUPlace(), false, dense_x.get());
  phi::Copy(*dev_ctx, *ref_y.get(), phi::GPUPlace(), false, dense_y.get());

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::matmul(x, y, false, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());

  auto ref_out = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, out.dims(), phi::DataLayout::NCHW));

  phi::Copy(*dev_ctx, *dense_out.get(), phi::CPUPlace(), false, ref_out.get());

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i], ref_out->data<float>()[i], 1e-6f);
  }
}

#endif

TEST(API, matmul_double_grad) {
  // 1. create tensor
  auto x = paddle::experimental::full({3, 3}, 1.0);
  auto y = paddle::experimental::full({3, 3}, 2.0);
  auto out_grad = paddle::experimental::full({3, 3}, 2.0);
  auto dx_grad = paddle::experimental::full({3, 3}, 2.0);

  // 2. test API
  const auto out = paddle::experimental::matmul_double_grad(
      x, y, out_grad, dx_grad, {}, false, false);

  // 3. check result
  ASSERT_EQ(out.size(), 3UL);
  ASSERT_EQ(out[0].size(), 1UL);
  ASSERT_EQ(out[1].size(), 1UL);
  ASSERT_EQ(out[2].size(), 1UL);
  ASSERT_EQ(out[0][0].dims()[1], 3);
  ASSERT_EQ(out[0][0].numel(), 9);
  ASSERT_EQ(out[1][0].numel(), 9);
  ASSERT_EQ(out[2][0].numel(), 9);
  ASSERT_EQ(out[0][0].type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out[0][0].layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out[1][0].initialized(), true);
  ASSERT_EQ(out[2][0].initialized(), true);
}

}  // namespace tests
}  // namespace paddle
