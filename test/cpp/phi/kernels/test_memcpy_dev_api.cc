// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/memcpy_kernel.h"

namespace phi {
namespace tests {

using DDim = phi::DDim;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(DEV_API, memcpy_d2h) {
  // 1. create tensor
  const auto cpu_alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor x_cpu(cpu_alloc.get(),
                         phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                              common::make_ddim({3, 2, 2, 3}),
                                              phi::DataLayout::NCHW));
  auto& pool = phi::DeviceContextPool::Instance();
  auto* cpu_ctx = pool.GetByPlace(phi::CPUPlace());
  auto* x_cpu_data = cpu_ctx->template Alloc<float>(&x_cpu);

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = static_cast<float>(i);
  }

  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  phi::DenseTensor x;

  // 2. test API
  auto* dev_ctx = pool.GetByPlace(phi::GPUPlace());

  phi::MemcpyH2DKernel<phi::GPUContext>(*dev_ctx, x_cpu, 1, &x);
  phi::DenseTensor out;
  phi::MemcpyD2HKernel<phi::GPUContext>(*dev_ctx, x, 1, &out);

  // 3. check result
  std::vector<int64_t> expect_shape = {12, 3};
  ASSERT_EQ(out.dims(), x.dims());
  ASSERT_EQ(out.meta().dtype, phi::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, phi::DataLayout::NCHW);

  bool value_equal = true;
  auto* dense_out_data = out.data<float>();
  for (int i = 0; i < x_cpu.numel(); i++) {
    if (x_cpu_data[i] != dense_out_data[i]) {
      value_equal = false;
      break;
    }
  }
  ASSERT_EQ(value_equal, true);
}

#endif
}  // namespace tests
}  // namespace phi
