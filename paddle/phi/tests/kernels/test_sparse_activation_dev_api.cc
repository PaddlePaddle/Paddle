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
#include "paddle/phi/kernels/activation_grad_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_activation_grad_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_activation_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace tests {

TEST(DEV_API, sparse_relu) {
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
  dev_ctx_cpu.Init();

  DenseTensor dense_x =
      phi::Empty(dev_ctx_cpu,
                 DenseTensorMeta(DataType::FLOAT32, {3, 4}, DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_coo = sparse::DenseToSparseCoo<float>(dev_ctx_cpu, dense_x, 2);

  auto sparse_out = sparse::SparseRelu<float>(dev_ctx_cpu, sparse_coo);
  DenseTensor dense_out =
      phi::EmptyLike<float>(dev_ctx_cpu, sparse_out.non_zero_elements());
  ReluKernel<float>(dev_ctx_cpu, sparse_coo.non_zero_elements(), &dense_out);

  int cmp = memcmp(dense_out.data<float>(),
                   sparse_out.non_zero_elements().data<float>(),
                   dense_out.numel() * sizeof(float));
  ASSERT_EQ(cmp, 0);
  // backward
  DenseTensor dense_grad_x = phi::EmptyLike<float>(dev_ctx_cpu, dense_out);
  ReluGradKernel<float>(
      dev_ctx_cpu, sparse_coo.non_zero_elements(), dense_out, &dense_grad_x);
  SparseCooTensor sparse_grad_x(
      phi::EmptyLike<int>(dev_ctx_cpu, sparse_coo.non_zero_indices()),
      phi::EmptyLike<int>(dev_ctx_cpu, sparse_coo.non_zero_elements()),
      {3, 4});

  SparseCooTensor sparse_out_grad(
      sparse_coo.non_zero_indices(), dense_out, {3, 4});
  sparse::SparseReluGradKernel<float>(
      dev_ctx_cpu, sparse_coo, sparse_out_grad, &sparse_grad_x);

  cmp = memcmp(dense_grad_x.data<float>(),
               sparse_grad_x.non_zero_elements().data<float>(),
               dense_grad_x.numel() * sizeof(float));
  ASSERT_EQ(cmp, 0);
}

}  // namespace tests
}  // namespace phi
