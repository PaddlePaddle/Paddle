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

#include <gtest/gtest.h>

#include <memory>

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"
// #include "paddle/phi/kernels/transpose_grad_kernel.h"
// #include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/reshape_grad_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace phi {
namespace tests {

TEST(DEV_API, sparse_reshape) {
  std::vector<float> data = {0, -1, 0, 2, 0, 0, -3, 0, 4, 5, 0, 0};
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());

  DenseTensor dense_x = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, {3, 2, 2}, DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_coo = sparse::DenseToCoo<float>(dev_ctx_cpu, dense_x, 3);
  auto sparse_out =
      sparse::ReshapeCoo<float>(dev_ctx_cpu, sparse_coo, {2L, 6L});
  DenseTensor dense_out = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, {2, 6}, DataLayout::NCHW));
  ReshapeKernel(dev_ctx_cpu, dense_x, {2, 6}, &dense_out);

  for (int i = 0; i < dense_out.numel(); ++i) {
    ASSERT_EQ(
        dense_out.data<float>()[i],
        sparse::CooToDense<float>(dev_ctx_cpu, sparse_out).data<float>()[i]);
  }
}

}  // namespace tests
}  // namespace phi