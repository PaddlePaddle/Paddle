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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/sparse_copy_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/sparse_coo_tensor.h"

namespace phi {
namespace tests {

TEST(DEV_API, copy_sparse_coo) {
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DDim dense_dims = {2, 2};
  //(0,0), (1,1)
  std::vector<int> indices = {0, 1, 0, 1};
  std::vector<float> elements = {1.0, 2.0};

  DenseTensor src_indices(
      alloc.get(), DenseTensorMeta(DataType::INT32, {2, 2}, DataLayout::NCHW));
  DenseTensor src_elements(
      alloc.get(), DenseTensorMeta(DataType::FLOAT32, {2}, DataLayout::NCHW));

  phi::CPUPlace cpu;
  memcpy(src_indices.mutable_data<int>(cpu),
         indices.data(),
         sizeof(int) * indices.size());
  memcpy(src_elements.mutable_data<int>(cpu),
         elements.data(),
         sizeof(float) * elements.size());
  SparseCooTensor src_coo(src_indices, src_elements, dense_dims);
  SparseCooTensor dst_coo;

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_cpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  sparse::CopyCoo(dev_ctx_cpu, src_coo, cpu, true, &dst_coo);
  int cmp_indices = memcmp(dst_coo.non_zero_indices().data(),
                           indices.data(),
                           sizeof(int) * indices.size());
  int cmp_elements = memcmp(dst_coo.non_zero_elements().data(),
                            elements.data(),
                            sizeof(float) * elements.size());
  ASSERT_EQ(cmp_indices, 0);
  ASSERT_EQ(cmp_elements, 0);
}

}  // namespace tests
}  // namespace phi
