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

#include "paddle/pten/api/include/utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

paddle::experimental::Tensor CreateInputTensor() {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::INT64,
                            framework::make_ddim({3, 4}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<int64_t>();

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
  ASSERT_EQ(out.type(), pten::DataType::INT64);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
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
  auto tmp = paddle::experimental::copy_to(x, pten::Backend::GPU, false);
  auto out = paddle::experimental::copy_to(tmp, pten::Backend::CPU, true);
#else
  auto out = paddle::experimental::copy_to(x, pten::Backend::CPU, false);
#endif

  // 3. check result
  CheckOutputResult(out);
}

TEST(Tensor, copy_to) {
  // 1. create tensor
  auto x = CreateInputTensor();

// 2. test API
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto tmp = x.copy_to(pten::Backend::GPU, false);
  auto out = tmp.copy_to(pten::Backend::CPU, true);
#else
  auto out = x.copy_to(pten::Backend::CPU, false);
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
