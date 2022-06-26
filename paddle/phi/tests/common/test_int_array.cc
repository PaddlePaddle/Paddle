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

#include "gtest/gtest.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
#endif

namespace phi {
namespace tests {

TEST(IntArray, ConstructFromCPUDenseTensor) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const auto* dev_ctx =
      static_cast<const phi::CPUContext*>(pool.Get(CPUPlace()));
  phi::DenseTensor shape = Full<int>(*dev_ctx, {2}, 3);
  phi::DenseTensor out = Full<int>(*dev_ctx, shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromCPUDenseTensorVector) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const auto* dev_ctx =
      static_cast<const phi::CPUContext*>(pool.Get(CPUPlace()));
  phi::DenseTensor shape0 = Full<int>(*dev_ctx, {1}, 3);
  phi::DenseTensor shape1 = Full<int64_t>(*dev_ctx, {1}, 3);
  std::vector<phi::DenseTensor> shape{shape0, shape1};
  phi::DenseTensor out = Full<int>(*dev_ctx, shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromCPUTensor) {
  auto shape = paddle::experimental::full({2}, 3, DataType::INT64);
  auto out = paddle::experimental::full(shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromCPUTensorVector) {
  auto shape0 = paddle::experimental::full({2}, 3, DataType::INT64);
  auto shape1 = paddle::experimental::full({2}, 3, DataType::INT32);

  std::vector<paddle::experimental::Tensor> shape{shape0, shape0};
  auto out = paddle::experimental::full(shape, 1);

  std::vector<paddle::experimental::Tensor> shape_new{shape0, shape1};
  auto out1 = paddle::experimental::full(shape_new, 1);

  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);

  ASSERT_EQ(out1.dims().size(), 2);
  ASSERT_EQ(out1.dims()[0], 3);
  ASSERT_EQ(out1.dims()[1], 3);
  ASSERT_EQ(out1.numel(), 9);
}

TEST(IntArray, ThrowException) {
  auto shape = paddle::experimental::full({2}, 3, DataType::FLOAT32);
  auto create_int_array = [&shape]() -> paddle::experimental::IntArray {
    paddle::experimental::IntArray int_array{shape};
    return int_array;
  };
  ASSERT_ANY_THROW(create_int_array());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(IntArray, ConstructFromGPUDenseTensor) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const auto* dev_ctx =
      static_cast<const phi::GPUContext*>(pool.Get(GPUPlace()));
  phi::DenseTensor shape = Full<int>(*dev_ctx, {2}, 3);
  phi::DenseTensor out = Full<int>(*dev_ctx, shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromGPUDenseTensorVector) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const auto* dev_ctx =
      static_cast<const phi::GPUContext*>(pool.Get(GPUPlace()));
  phi::DenseTensor shape0 = Full<int>(*dev_ctx, {1}, 3);
  phi::DenseTensor shape1 = Full<int64_t>(*dev_ctx, {1}, 3);
  std::vector<phi::DenseTensor> shape{shape0, shape1};
  phi::DenseTensor out = Full<int>(*dev_ctx, shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromGPUTensor) {
  auto shape = paddle::experimental::full({2}, 3, DataType::INT64, GPUPlace());
  auto out = paddle::experimental::full(shape, 1);
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
}

TEST(IntArray, ConstructFromGPUTensorVector) {
  auto shape0 = paddle::experimental::full({2}, 3, DataType::INT64, GPUPlace());
  auto shape1 = paddle::experimental::full({2}, 3, DataType::INT32, GPUPlace());

  std::vector<paddle::experimental::Tensor> shape{shape0, shape0};
  auto out = paddle::experimental::full(shape, 1);

  std::vector<paddle::experimental::Tensor> shape_new{shape0, shape1};
  auto out1 = paddle::experimental::full(shape_new, 1);

  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);

  ASSERT_EQ(out1.dims().size(), 2);
  ASSERT_EQ(out1.dims()[0], 3);
  ASSERT_EQ(out1.dims()[1], 3);
  ASSERT_EQ(out1.numel(), 9);
}
#endif

}  // namespace tests
}  // namespace phi
