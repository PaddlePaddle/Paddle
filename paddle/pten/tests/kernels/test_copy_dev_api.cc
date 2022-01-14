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

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/copy_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(YuanRisheng): This TEST file need to be refactored after 'copy' realized
// in 'paddle/api'
TEST(DEV_API, copy) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_src = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({2, 3}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_src->mutable_data<float>();

  auto dense_dst = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({2, 3}),
                            pten::DataLayout::NCHW));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dense_x_data[i * 3 + j] = (i * 3 + j) * 1.0;
    }
  }
  const auto& a = paddle::platform::CPUPlace();
  std::cout << typeid(a).name() << std::endl;
  // 2. test API
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(paddle::platform::CPUPlace());
  pten::Copy(*dev_ctx, *(dense_src.get()), false, dense_dst.get());

  // 3. check result
  for (int64_t i = 0; i < dense_src->numel(); i++) {
    ASSERT_EQ(dense_src->data<float>()[i], dense_dst->data<float>()[i]);
  }
}

}  // namespace tests
}  // namespace pten
