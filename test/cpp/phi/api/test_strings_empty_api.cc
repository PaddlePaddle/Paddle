/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include <memory>

#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/string_tensor.h"

PD_DECLARE_KERNEL(strings_empty, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(strings_empty_like, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

using phi::CPUPlace;
using phi::StringTensor;
using phi::StringTensorMeta;

TEST(API, strings_empty) {
  // 1. create tensor
  auto cpu = CPUPlace();
  const auto alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(cpu);

  auto dense_shape = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT64, common::make_ddim({2}), phi::DataLayout::NCHW));
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());
  auto* shape_data = dev_ctx->template Alloc<int64_t>(dense_shape.get());

  shape_data[0] = 2;
  shape_data[1] = 3;

  paddle::Tensor tensor_shape(dense_shape);

  // 2. test API
  auto empty_out = paddle::experimental::strings::empty(tensor_shape);

  // 3. check result
  ASSERT_EQ(empty_out.dims().size(), 2);
  ASSERT_EQ(empty_out.dims()[0], 2);
  ASSERT_EQ(empty_out.dims()[1], 3);
  ASSERT_EQ(empty_out.numel(), 6);
}

TEST(API, strings_empty_like) {
  auto cpu = CPUPlace();
  const auto alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(cpu);
  // 1. create tensor
  const phi::DDim dims({1, 2});
  StringTensorMeta meta(dims);
  auto cpu_strings_x = std::make_shared<phi::StringTensor>(
      alloc.get(), phi::StringTensorMeta(meta));

  // 2. test API
  paddle::Tensor x(cpu_strings_x);
  auto empty_like_out = paddle::experimental::strings::empty_like(x);

  // 3. check result
  ASSERT_EQ(empty_like_out.dims().size(), 2);
  ASSERT_EQ(empty_like_out.dims()[0], 1);
  ASSERT_EQ(empty_like_out.numel(), 2);
}

}  // namespace tests
}  // namespace paddle
