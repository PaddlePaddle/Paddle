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

#include "paddle/phi/api/include/api.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(copy, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

paddle::experimental::Tensor CreateInputTensor() {
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT64, phi::make_ddim({3, 4}), phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<int64_t>(paddle::platform::CPUPlace());

  for (int64_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i;
  }

  return paddle::experimental::Tensor(dense_x);
}

void CheckOutputResult(const paddle::experimental::Tensor& out) {
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 4);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::INT64);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  for (int64_t i = 0; i < 12; ++i) {
    ASSERT_EQ(out.data<int64_t>()[i], i);
  }
}

TEST(API, copy_to) {
  // 1. create tensor
  auto x = CreateInputTensor();

// 2. test API
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto tmp = paddle::experimental::copy_to(x, phi::GPUPlace(), false);
  auto out = paddle::experimental::copy_to(tmp, phi::CPUPlace(), true);
#else
  auto out = paddle::experimental::copy_to(x, phi::CPUPlace(), false);
#endif

  // 3. check result
  CheckOutputResult(out);
}

TEST(Tensor, copy_to) {
  // 1. create tensor
  auto x = CreateInputTensor();

// 2. test API
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto tmp = x.copy_to(phi::GPUPlace(), false);
  auto out = tmp.copy_to(phi::CPUPlace(), true);
#else
  auto out = x.copy_to(phi::CPUPlace(), false);
#endif

  // 3. check result
  CheckOutputResult(out);
}

TEST(Tensor, old_copy_to) {
  // 1. create tensor
  auto x = CreateInputTensor();

// 2. test API
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto tmp = x.copy_to<int64_t>(paddle::PlaceType::kGPU);
  auto out = tmp.copy_to<int64_t>(paddle::PlaceType::kCPU);
#else
  auto out = x.copy_to<int64_t>(paddle::PlaceType::kCPU);
#endif

  // 3. check result
  CheckOutputResult(out);
}

}  // namespace tests
}  // namespace paddle
