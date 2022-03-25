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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/sparse_activation_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace tests {

TEST(DEV_API, sparse_relu) {
  std::vector<float> data = {0, -1, 0, 2, 0, 0, -3, 0, 4, 5, 0, 0};
  std::vector<float> correct_out = {0, 2, 0, 4, 5};
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  DenseTensor dense_x =
      phi::Empty(dev_ctx_cpu,
                 DenseTensorMeta(DataType::FLOAT32, {3, 4}, DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_coo = sparse::DenseToSparseCoo<float>(dev_ctx_cpu, dense_x, 2);

  auto act_out = sparse::SparseRelu<float>(dev_ctx_cpu, sparse_coo);

  int cmp = memcmp(correct_out.data(),
                   act_out.non_zero_elements().data<float>(),
                   correct_out.size() * sizeof(float));
  ASSERT_EQ(cmp, 0);
}

}  // namespace tests
}  // namespace phi
