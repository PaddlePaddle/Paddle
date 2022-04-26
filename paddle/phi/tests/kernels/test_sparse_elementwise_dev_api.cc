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

#include <cmath>
#include <memory>

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_grad_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace tests {

#define TEST_ELEMENTWISE_OP(name)          \
  TEST_ELEMENTWISE_OP_WITH_TYPE(name, Csr) \
                                           \
  TEST_ELEMENTWISE_OP_WITH_TYPE(name, Coo)

#define TEST_ELEMENTWISE_OP_WITH_TYPE(name, type)                     \
  template <typename T, typename Context>                             \
  void TestElementWise##name##type(const Context& dev_ctx_cpu,        \
                                   const Sparse##type##Tensor& x,     \
                                   const Sparse##type##Tensor& y,     \
                                   const DDim& dense_dims) {          \
    auto out = sparse::ElementWise##name##type<T>(dev_ctx_cpu, x, y); \
    const DenseTensor denseX =                                        \
        sparse::Sparse##type##ToDense<T>(dev_ctx_cpu, x);             \
    const DenseTensor denseY =                                        \
        sparse::Sparse##type##ToDense<T>(dev_ctx_cpu, y);             \
    const DenseTensor denseOut =                                      \
        sparse::Sparse##type##ToDense<T>(dev_ctx_cpu, out);           \
                                                                      \
    auto expectResult = name<T>(dev_ctx_cpu, denseX, denseY);         \
    for (int j = 0; j < denseOut.numel(); ++j) {                      \
      auto actualResultRow = denseOut.template data<T>()[j];          \
      auto expectResultRow = expectResult.template data<T>()[j];      \
      if (!std::isnan(expectResultRow)) {                             \
        ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);           \
      }                                                               \
    }                                                                 \
  }

TEST_ELEMENTWISE_OP(Add)
TEST_ELEMENTWISE_OP(Subtract)
TEST_ELEMENTWISE_OP(Multiply)
TEST_ELEMENTWISE_OP(Divide)

TEST(DEV_API, sparse_elementwise_coo_kernel_double) {
  using T = double;
  DDim dense_dims = phi::make_ddim({2, 4, 4});

  std::vector<T> x_dense_data = {0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0,
                                 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0,
                                 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0};

  std::vector<T> y_dense_data = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0,
                                 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<T>(paddle::platform::CPUPlace());

  memcpy(dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(T));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data = dense_y.mutable_data<T>(paddle::platform::CPUPlace());

  memcpy(dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(T));

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  auto coo_x = sparse::DenseToSparseCoo<T>(dev_ctx_cpu, dense_x, 2);
  auto coo_y = sparse::DenseToSparseCoo<T>(dev_ctx_cpu, dense_y, 2);

  TestElementWiseAddCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
  TestElementWiseSubtractCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
  TestElementWiseMultiplyCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
  TestElementWiseDivideCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
}

TEST(DEV_API, sparse_elementwise_op_csr_kernel_float) {
  using T = float;
  DDim dense_dims = phi::make_ddim({2, 3, 4});

  std::vector<T> x_dense_data = {0.0, 0.0, 4.0, 2.0, 6.0, 3.0, 0.2, 0.1,
                                 2.2, 1.1, 4.2, 2.1, 0.4, 0.2, 0.0, 0.0,
                                 4.4, 2.2, 0.6, 0.3, 2.6, 1.3, 0.0, 0.0};
  std::vector<T> y_dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5,
                                 0.7, 0.0, 3.5, 0.7, 3.2, 0.1, 0.0, 3.2,
                                 1.0, 0.0, 1.2, 0.5, 0.7, 3.3, 0.0, 9.0};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<T>(paddle::platform::CPUPlace());

  memcpy(dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(T));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data = dense_y.mutable_data<T>(paddle::platform::CPUPlace());

  memcpy(dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(T));

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  auto csr_x = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_y);

  TestElementWiseAddCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

#define TEST_ELEMENTWISE_GRAD(name)                                         \
  template <typename T, typename Context>                                   \
  void TestElementWise##name##Grad(const Context& dev_ctx_cpu,              \
                                   const SparseCsrTensor& x,                \
                                   const SparseCsrTensor& y,                \
                                   const DDim& dense_dims) {                \
    auto out = sparse::ElementWise##name##Csr<T>(dev_ctx_cpu, x, y);        \
    auto dresult =                                                          \
        sparse::ElementWise##name##CsrGrad<T>(dev_ctx_cpu, x, y, out);      \
                                                                            \
    DenseTensor expectdy = phi::Empty(                                      \
        dev_ctx_cpu,                                                        \
        DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));  \
    DenseTensor expectdx = phi::Empty(                                      \
        dev_ctx_cpu,                                                        \
        DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));  \
                                                                            \
    phi::name##GradKernel<T>(dev_ctx_cpu,                                   \
                             sparse::SparseCsrToDense<T>(dev_ctx_cpu, x),   \
                             sparse::SparseCsrToDense<T>(dev_ctx_cpu, y),   \
                             sparse::SparseCsrToDense<T>(dev_ctx_cpu, out), \
                             -1,                                            \
                             &expectdx,                                     \
                             &expectdy);                                    \
    const DenseTensor densedX =                                             \
        sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);               \
    const DenseTensor densedY =                                             \
        sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);               \
    const DenseTensor denseOut =                                            \
        sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);                      \
                                                                            \
    for (int j = 0; j < densedX.numel(); ++j) {                             \
      auto actualResultRow = densedX.template data<T>()[j];                 \
      auto expectResultRow = expectdx.template data<T>()[j];                \
      ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);                 \
    }                                                                       \
    for (int j = 0; j < densedY.numel(); ++j) {                             \
      auto actualResultRow = densedY.template data<T>()[j];                 \
      auto expectResultRow = expectdy.template data<T>()[j];                \
      ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);                 \
    }                                                                       \
  }

TEST_ELEMENTWISE_GRAD(Add)
TEST_ELEMENTWISE_GRAD(Subtract)
TEST_ELEMENTWISE_GRAD(Multiply)

template <typename T, typename Context>
void TestElementWiseDivideGrad(const Context& dev_ctx_cpu,
                               const SparseCsrTensor& x,
                               const SparseCsrTensor& y,
                               const DDim& dense_dims) {
  auto out = sparse::ElementWiseDivideCsr<T>(dev_ctx_cpu, x, y);
  auto dresult =
      sparse::ElementWiseDivideCsrGrad<T>(dev_ctx_cpu, x, y, out, out);
  DenseTensor expectdy = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  DenseTensor expectdx = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  phi::DivideGradKernel<T>(dev_ctx_cpu,
                           sparse::SparseCsrToDense<T>(dev_ctx_cpu, x),
                           sparse::SparseCsrToDense<T>(dev_ctx_cpu, y),
                           sparse::SparseCsrToDense<T>(dev_ctx_cpu, out),
                           sparse::SparseCsrToDense<T>(dev_ctx_cpu, out),
                           -1,
                           &expectdx,
                           &expectdy);
  const DenseTensor densedX =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  for (int j = 0; j < densedX.numel(); ++j) {
    auto actualResultRow = densedX.template data<T>()[j];
    auto expectResultRow = expectdx.template data<T>()[j];
    if (!std::isnan(expectResultRow)) {
      ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);
    }
  }
  for (int j = 0; j < densedY.numel(); ++j) {
    auto actualResultRow = densedY.template data<T>()[j];
    auto expectResultRow = expectdy.template data<T>()[j];
    if (!std::isnan(expectResultRow)) {
      ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);
    }
  }
}

TEST(DEV_API, sparse_elementwise_op_csr_grad_kernel_float) {
  using T = float;
  DDim dense_dims = phi::make_ddim({2, 3, 4});

  std::vector<T> x_dense_data = {0.0, 0.0, 4.0, 2.0, 6.0, 3.0, 0.2, 0.1,
                                 2.2, 1.1, 4.2, 2.1, 0.4, 0.2, 0.0, 0.0,
                                 4.4, 2.2, 0.6, 0.3, 2.6, 1.3, 0.0, 0.0};
  std::vector<T> y_dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5,
                                 0.7, 0.0, 3.5, 0.7, 3.2, 0.1, 0.0, 3.2,
                                 1.0, 0.0, 1.2, 0.5, 0.7, 3.3, 0.0, 9.0};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<T>(paddle::platform::CPUPlace());
  memcpy(dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(T));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data = dense_y.mutable_data<T>(paddle::platform::CPUPlace());
  memcpy(dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(T));

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

  auto csr_x = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_y);

  auto dx = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_y);
  auto dy = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);

  TestElementWiseAddGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

TEST(DEV_API, sparse_elementwise_op_csr_grad_kernel_double) {
  using T = double;
  DDim dense_dims = phi::make_ddim({1, 12});
  std::vector<T> x_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0, 3.2, 0.0, 0.0};
  std::vector<T> y_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5, 0.7, 0.0, 3.5, 0.7};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<T>(paddle::platform::CPUPlace());
  memcpy(dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(T));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data = dense_y.mutable_data<T>(paddle::platform::CPUPlace());
  memcpy(dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(T));

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

  auto csr_x = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_y);

  auto dx = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_y);
  auto dy = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);

  TestElementWiseAddGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

}  // namespace tests
}  // namespace phi
