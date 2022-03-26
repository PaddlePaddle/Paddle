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

#include <map>  // NOLINT
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace tests {

using DDim = phi::DDim;

__global__ void FillTensor(float* data) { data[0] = 1; }

TEST(Scalar, ConstructFromDenseTensor) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::GPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::GPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<float>(&dense_x);
  FillTensor<<<1, 1, 0, dev_ctx.stream()>>>(dense_x_data);
  dev_ctx.Wait();
  phi::Scalar scalar_test(dense_x);
  ASSERT_NEAR(1, scalar_test.to<float>(), 1e-6);
}

}  // namespace tests
}  // namespace phi
