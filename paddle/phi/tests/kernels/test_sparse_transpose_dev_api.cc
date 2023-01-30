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
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"
#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
namespace phi {
namespace tests {

TEST(DEV_API, sparse_transpose_coo) {
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
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({3, 2, 2}), DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_coo = sparse::DenseToCoo<float>(dev_ctx_cpu, dense_x, 3);
  auto sparse_out =
      sparse::TransposeCoo<float>(dev_ctx_cpu, sparse_coo, {2, 1, 0});
  DenseTensor dense_out = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({2, 2, 3}), DataLayout::NCHW));
  TransposeKernel<float>(dev_ctx_cpu, dense_x, {2, 1, 0}, &dense_out);

  // backward
  DenseTensor dense_grad_x = phi::EmptyLike<float>(dev_ctx_cpu, dense_out);
  TransposeGradKernel<float>(dev_ctx_cpu, dense_out, {2, 1, 0}, &dense_grad_x);
  SparseCooTensor sparse_grad_x;
  sparse::EmptyLikeCooKernel<float>(dev_ctx_cpu, sparse_coo, &sparse_grad_x);

  SparseCooTensor sparse_out_grad(
      sparse_coo.indices(), sparse_coo.values(), {2, 2, 3});
  sparse::TransposeCooGradKernel<float>(
      dev_ctx_cpu, sparse_out_grad, {2, 1, 0}, &sparse_grad_x);
}

TEST(DEV_API, sparse_transpose_csr_case1) {
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
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({3, 2, 2}), DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_csr = sparse::DenseToCsr<float>(dev_ctx_cpu, dense_x);

  auto sparse_out =
      sparse::TransposeCsr<float>(dev_ctx_cpu, sparse_csr, {2, 1, 0});
  DenseTensor dense_out = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({2, 2, 3}), DataLayout::NCHW));
  TransposeKernel<float>(dev_ctx_cpu, dense_x, {2, 1, 0}, &dense_out);

  // backward
  DenseTensor dense_grad_x = phi::EmptyLike<float>(dev_ctx_cpu, dense_out);
  TransposeGradKernel<float>(dev_ctx_cpu, dense_out, {2, 1, 0}, &dense_grad_x);
  SparseCsrTensor sparse_grad_x;
  sparse::EmptyLikeCsrKernel<float>(dev_ctx_cpu, sparse_csr, &sparse_grad_x);
  sparse::TransposeCsrGradKernel<float>(
      dev_ctx_cpu, sparse_out, {2, 1, 0}, &sparse_grad_x);
}

TEST(DEV_API, sparse_transpose_csr_case2) {
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
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({3, 2, 2}), DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_csr = sparse::DenseToCsr<float>(dev_ctx_cpu, dense_x);

  auto sparse_out =
      sparse::TransposeCsr<float>(dev_ctx_cpu, sparse_csr, {1, 2, 0});
  DenseTensor dense_out = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({2, 2, 3}), DataLayout::NCHW));
  TransposeKernel<float>(dev_ctx_cpu, dense_x, {1, 2, 0}, &dense_out);
}

TEST(DEV_API, sparse_transpose_csr_case3) {
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
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({3, 4}), DataLayout::NCHW));
  memcpy(dense_x.data<float>(), data.data(), data.size() * sizeof(float));
  auto sparse_csr = sparse::DenseToCsr<float>(dev_ctx_cpu, dense_x);

  auto sparse_out =
      sparse::TransposeCsr<float>(dev_ctx_cpu, sparse_csr, {1, 0});
  DenseTensor dense_out = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(
          DataType::FLOAT32, phi::make_ddim({4, 3}), DataLayout::NCHW));
  TransposeKernel<float>(dev_ctx_cpu, dense_x, {1, 0}, &dense_out);
}

}  // namespace tests
}  // namespace phi
