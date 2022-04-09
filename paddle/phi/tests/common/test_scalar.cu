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
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);

namespace phi {
namespace tests {

using DDim = phi::DDim;
using float16 = phi::dtype::float16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

__global__ void FillTensor(float* data) { data[0] = 1; }

TEST(Scalar, ConstructFromDenseTensor1) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT16, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<float16>(&dense_x);
  dense_x_data[0] = 1;
  phi::Scalar scalar_test(dense_x);
  ASSERT_NEAR(1, scalar_test.to<float16>(), 1e-6);
}

TEST(Scalar, ConstructFromDenseTensor2) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT16, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<int16_t>(&dense_x);
  dense_x_data[0] = 1;
  phi::Scalar scalar_test(dense_x);
  ASSERT_EQ(1, scalar_test.to<int16_t>());
}

TEST(Scalar, ConstructFromDenseTensor3) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT8, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<int8_t>(&dense_x);
  dense_x_data[0] = 1;
  phi::Scalar scalar_test(dense_x);
  ASSERT_EQ(1, scalar_test.to<int8_t>());
}

TEST(Scalar, ConstructFromDenseTensor4) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::BOOL, phi::make_ddim({1}), phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<bool>(&dense_x);
  dense_x_data[0] = true;
  phi::Scalar scalar_test(dense_x);
  ASSERT_EQ(true, scalar_test.to<bool>());
}

TEST(Scalar, ConstructFromDenseTensor5) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(alloc.get(),
                           phi::DenseTensorMeta(phi::DataType::COMPLEX64,
                                                phi::make_ddim({1}),
                                                phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<complex64>(&dense_x);
  dense_x_data[0] = 1;
  phi::Scalar scalar_test(dense_x);
  complex64 expected_value(1, 0);
  EXPECT_TRUE(expected_value == scalar_test.to<complex64>());
}

TEST(Scalar, ConstructFromDenseTensor6) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  phi::DenseTensor dense_x(alloc.get(),
                           phi::DenseTensorMeta(phi::DataType::COMPLEX128,
                                                phi::make_ddim({1}),
                                                phi::DataLayout::NCHW));
  phi::CPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::CPUPlace())
                           .get());
  dev_ctx.Init();

  auto* dense_x_data = dev_ctx.Alloc<complex128>(&dense_x);
  dense_x_data[0] = 1;
  phi::Scalar scalar_test(dense_x);
  complex128 expected_value(1, 0);
  EXPECT_TRUE(expected_value == scalar_test.to<complex128>());
}

TEST(Scalar, ConstructFromDenseTensor7) {
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

TEST(Scalar, ConstructFromTensor) {
  // 1. create tensor
  const auto alloc =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({1}), phi::DataLayout::NCHW));

  phi::GPUContext dev_ctx;
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::GPUPlace())
                           .get());
  dev_ctx.Init();
  auto* dense_x_data = dev_ctx.Alloc<float>(dense_x.get());
  FillTensor<<<1, 1, 0, dev_ctx.stream()>>>(dense_x_data);
  dev_ctx.Wait();
  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Scalar scalar_test(x);
  ASSERT_NEAR(1, scalar_test.to<float>(), 1e-6);
}

}  // namespace tests
}  // namespace phi
