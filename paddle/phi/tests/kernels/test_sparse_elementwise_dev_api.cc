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
#include "paddle/phi/kernels/sparse/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/sparse/elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace tests {

#define TEST_ELEMENTWISE_OP(name)          \
  TEST_ELEMENTWISE_OP_WITH_TYPE(name, Csr) \
                                           \
  TEST_ELEMENTWISE_OP_WITH_TYPE(name, Coo)

#define TEST_ELEMENTWISE_OP_WITH_TYPE(name, type)                            \
  template <typename T, typename Context>                                    \
  void TestElementWise##name##type(const Context& dev_ctx_cpu,               \
                                   const Sparse##type##Tensor& x,            \
                                   const Sparse##type##Tensor& y,            \
                                   const DDim& dense_dims) {                 \
    auto out = sparse::ElementWise##name##type<T>(dev_ctx_cpu, x, y);        \
    const DenseTensor denseX = sparse::type##ToDense<T>(dev_ctx_cpu, x);     \
    const DenseTensor denseY = sparse::type##ToDense<T>(dev_ctx_cpu, y);     \
    const DenseTensor denseOut = sparse::type##ToDense<T>(dev_ctx_cpu, out); \
    auto expectResult = name<T>(dev_ctx_cpu, denseX, denseY);                \
    for (int j = 0; j < denseOut.numel(); ++j) {                             \
      auto actualResultRow = denseOut.template data<T>()[j];                 \
      auto expectResultRow = expectResult.template data<T>()[j];             \
      if (std::is_same<T, float>::value || std::is_same<T, double>::value) { \
        if (!std::isnan(expectResultRow)) {                                  \
          ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);                \
        }                                                                    \
      } else {                                                               \
        ASSERT_EQ(expectResultRow, actualResultRow);                         \
      }                                                                      \
    }                                                                        \
  }

TEST_ELEMENTWISE_OP(Add)
TEST_ELEMENTWISE_OP(Subtract)
TEST_ELEMENTWISE_OP(Multiply)
TEST_ELEMENTWISE_OP(Divide)

TEST(DEV_API, sparse_elementwise_coo_kernel_double) {
  using T = double;
  using IntT = int64_t;
  for (int epoch = 0; epoch < 100; ++epoch) {
    DDim dense_dims = phi::make_ddim({2, 4, 4});
    IntT sparse_dim = 2;
    // 32els
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

    auto coo_x = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_x, sparse_dim);
    auto coo_y = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_y, sparse_dim);

    TestElementWiseAddCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
    TestElementWiseSubtractCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
    TestElementWiseMultiplyCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
    TestElementWiseDivideCoo<T>(dev_ctx_cpu, coo_x, coo_y, dense_dims);
  }
}

TEST(DEV_API, sparse_elementwise_csr_kernel_float) {
  using T = float;

  DDim dense_dims = phi::make_ddim({6, 4});
  // 24els
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

  auto csr_x = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_y);

  TestElementWiseAddCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideCsr<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

#define TEST_ELEMENTWISE_OP_GRAD(name)          \
  TEST_ELEMENTWISE_OP_GRAD_WITH_TYPE(name, Csr) \
                                                \
  TEST_ELEMENTWISE_OP_GRAD_WITH_TYPE(name, Coo)

#define TEST_ELEMENTWISE_OP_GRAD_WITH_TYPE(name, type)                       \
  template <typename T, typename Context>                                    \
  void TestElementWise##name##type##Grad(const Context& dev_ctx_cpu,         \
                                         const Sparse##type##Tensor& x,      \
                                         const Sparse##type##Tensor& y,      \
                                         const DDim& dense_dims) {           \
    auto out = sparse::ElementWise##name##type<T>(dev_ctx_cpu, x, y);        \
    auto dresult =                                                           \
        sparse::ElementWise##name##type##Grad<T>(dev_ctx_cpu, x, y, out);    \
                                                                             \
    DenseTensor expectdy = phi::Empty(                                       \
        dev_ctx_cpu,                                                         \
        DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));   \
    DenseTensor expectdx = phi::Empty(                                       \
        dev_ctx_cpu,                                                         \
        DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));   \
                                                                             \
    phi::name##GradKernel<T>(dev_ctx_cpu,                                    \
                             sparse::type##ToDense<T>(dev_ctx_cpu, x),       \
                             sparse::type##ToDense<T>(dev_ctx_cpu, y),       \
                             sparse::type##ToDense<T>(dev_ctx_cpu, out),     \
                             -1,                                             \
                             &expectdx,                                      \
                             &expectdy);                                     \
    const DenseTensor densedX =                                              \
        sparse::type##ToDense<T>(dev_ctx_cpu, dresult[0]);                   \
    const DenseTensor densedY =                                              \
        sparse::type##ToDense<T>(dev_ctx_cpu, dresult[1]);                   \
    const DenseTensor denseOut = sparse::type##ToDense<T>(dev_ctx_cpu, out); \
                                                                             \
    for (int j = 0; j < densedX.numel(); ++j) {                              \
      auto actualResultRow = densedX.template data<T>()[j];                  \
      auto expectResultRow = expectdx.template data<T>()[j];                 \
      if (std::is_same<T, float>::value || std::is_same<T, double>::value) { \
        if (!std::isnan(expectResultRow)) {                                  \
          ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);                \
        }                                                                    \
      } else {                                                               \
        ASSERT_EQ(expectResultRow, actualResultRow);                         \
      }                                                                      \
    }                                                                        \
    for (int j = 0; j < densedY.numel(); ++j) {                              \
      auto actualResultRow = densedY.template data<T>()[j];                  \
      auto expectResultRow = expectdy.template data<T>()[j];                 \
      if (std::is_same<T, float>::value || std::is_same<T, double>::value) { \
        if (!std::isnan(expectResultRow)) {                                  \
          ASSERT_DOUBLE_EQ(expectResultRow, actualResultRow);                \
        }                                                                    \
      } else {                                                               \
        ASSERT_EQ(expectResultRow, actualResultRow);                         \
      }                                                                      \
    }                                                                        \
  }

TEST_ELEMENTWISE_OP_GRAD(Add)
TEST_ELEMENTWISE_OP_GRAD(Subtract)
TEST_ELEMENTWISE_OP_GRAD(Multiply)

template <typename T, typename Context>
void TestElementWiseDivideCsrGrad(const Context& dev_ctx_cpu,
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
                           sparse::CsrToDense<T>(dev_ctx_cpu, x),
                           sparse::CsrToDense<T>(dev_ctx_cpu, y),
                           sparse::CsrToDense<T>(dev_ctx_cpu, out),
                           sparse::CsrToDense<T>(dev_ctx_cpu, out),
                           -1,
                           &expectdx,
                           &expectdy);
  const DenseTensor densedX = sparse::CsrToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY = sparse::CsrToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::CsrToDense<T>(dev_ctx_cpu, out);
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

template <typename T, typename Context>
void TestElementWiseDivideCooGrad(const Context& dev_ctx_cpu,
                                  const SparseCooTensor& x,
                                  const SparseCooTensor& y,
                                  const DDim& dense_dims) {
  auto out = sparse::ElementWiseDivideCoo<T>(dev_ctx_cpu, x, y);
  auto dresult =
      sparse::ElementWiseDivideCooGrad<T>(dev_ctx_cpu, x, y, out, out);
  DenseTensor expectdy = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  DenseTensor expectdx = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  phi::DivideGradKernel<T>(dev_ctx_cpu,
                           sparse::CooToDense<T>(dev_ctx_cpu, x),
                           sparse::CooToDense<T>(dev_ctx_cpu, y),
                           sparse::CooToDense<T>(dev_ctx_cpu, out),
                           sparse::CooToDense<T>(dev_ctx_cpu, out),
                           -1,
                           &expectdx,
                           &expectdy);
  const DenseTensor densedX = sparse::CooToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY = sparse::CooToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::CooToDense<T>(dev_ctx_cpu, out);
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

TEST(DEV_API, sparse_elementwise_csr_grad_kernel_float) {
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

  auto csr_x = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_y);

  auto dx = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_y);
  auto dy = sparse::DenseToCsr<T>(dev_ctx_cpu, dense_x);

  TestElementWiseAddCsrGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractCsrGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyCsrGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideCsrGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

TEST(DEV_API, sparse_elementwise_coo_grad_kernel_double) {
  using T = double;
  int64_t sparse_dim = 2;
  DDim dense_dims = phi::make_ddim({3, 4});
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

  auto csr_x = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_x, sparse_dim);
  auto csr_y = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_y, sparse_dim);

  auto dx = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_y, sparse_dim);
  auto dy = sparse::DenseToCoo<T>(dev_ctx_cpu, dense_x, sparse_dim);

  TestElementWiseAddCooGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractCooGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyCooGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseDivideCooGrad<T>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

}  // namespace tests
}  // namespace phi
