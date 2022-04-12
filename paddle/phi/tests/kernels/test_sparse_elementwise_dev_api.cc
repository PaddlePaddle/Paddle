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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/copy_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_grad_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace tests {

#define TEST_ELEMENTWISE(name)                                              \
  template <typename T, typename Context>                                   \
  void TestElementWise##name##Csr(const Context& dev_ctx_cpu,               \
                                  const SparseCsrTensor& x,                 \
                                  const SparseCsrTensor& y,                 \
                                  const DDim& dense_dims) {                 \
    auto out = sparse::ElementWise##name##Csr<T>(dev_ctx_cpu, x, y);        \
    const DenseTensor denseX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, x); \
    const DenseTensor denseY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, y); \
    const DenseTensor denseOut =                                            \
        sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);                      \
                                                                            \
    auto expectResult = name<T>(dev_ctx_cpu, denseX, denseY);               \
                                                                            \
    for (int j = 0; j < denseOut.numel(); ++j) {                            \
      auto actualResultRow = denseOut.template data<T>()[j];                \
      auto expectResultRow = expectResult.template data<T>()[j];            \
      ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);                 \
    }                                                                       \
  }

TEST_ELEMENTWISE(Add)
TEST_ELEMENTWISE(Subtract)
TEST_ELEMENTWISE(Multiply)
// TEST_ELEMENTWISE(Divide)
/*template <typename T, typename Context>
void TestElementWiseDivideCsr(const Context& dev_ctx_cpu,
                              const SparseCsrTensor& x,
                              const SparseCsrTensor& y,
                              const DDim& dense_dims) {
  auto out = sparse::ElementWiseDivideCsr<T>(dev_ctx_cpu, x, y);
  const DenseTensor denseX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, x);
  const DenseTensor denseY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, y);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  auto expectResult = Divide<T>(dev_ctx_cpu, denseX, denseY);
  for (int j = 0; j < denseX.numel(); ++j) {
    auto actualResultRow = denseOut.data<T>()[j];
    std::cout << "actualResultRow: " << actualResultRow << std::endl;
    auto expectResultRow = expectResult.template data<T>()[j];
    std::cout << "expectResultRow: " << expectResultRow << std::endl;
  }
}*/

/*
template <typename T, typename Context>
void TestElementWiseAddCsr(const Context& dev_ctx_cpu,
                           const SparseCsrTensor& x,
                           const SparseCsrTensor& y,
                           const DDim& dense_dims) {
  auto out = sparse::ElementWiseAddCsr<T>(dev_ctx_cpu, x, y);
  const DenseTensor denseX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, x);
  const DenseTensor denseY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, y);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);

  auto expectResult = Add<T>(dev_ctx_cpu, denseX, denseY);

  for (int j = 0; j < denseOut.numel(); ++j) {
    auto actualResultRow = denseOut.template data<T>()[j];
    //    std::cout << "actualResultRow: " << actualResultRow << std::endl;
    auto expectResultRow = expectResult.template data<T>()[j];
    //    std::cout << "expectResultRow: " << expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}
*/

/*
template <typename T, typename Context>
void TestElementWiseSubtractCsr(const Context& dev_ctx_cpu,
                                const SparseCsrTensor& x,
                                const SparseCsrTensor& y,
                                const DDim& dense_dims) {
  auto out = sparse::ElementWiseSubtractCsr<T>(dev_ctx_cpu, x, y);
  const DenseTensor denseX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, x);
  const DenseTensor denseY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, y);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);

  auto expectResult = Subtract<T>(dev_ctx_cpu, denseX, denseY);

  for (int j = 0; j < denseOut.numel(); ++j) {
    auto actualResultRow = denseOut.template data<T>()[j];
    //    std::cout << "actualResultRow: " << actualResultRow << std::endl;
    auto expectResultRow = expectResult.template data<T>()[j];
    //    std::cout << "expectResultRow: " << expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}
*/

TEST(DEV_API, sparse_elementwise_op_csr_kernel) {
  DDim dense_dims = phi::make_ddim({4, 3});
  std::vector<float> x_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0, 3.2, 0.0, 0.0};
  /*  //  std::vector<float> x_non_zero_data = {1.0, 2.0, 3.0, 3.2};
    //  std::vector<int64_t> x_cols_data = {1, 0, 2, 0};
    //  std::vector<int64_t> x_crows_data = {0, 1, 3, 4};
    //  const int64_t x_non_zero_num = 4;*/

  std::vector<float> y_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5, 0.7, 0.0, 3.5, 0.7};

  //  std::vector<float> out_dense_data = {
  //      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.2, 3.5, 0.7};
  /*  //  std::vector<float> y_non_zero_data = {1.0, 2.0, 3.0, 3.5, 0.7};
    //  std::vector<int64_t> y_cols_data = {1, 0, 2, 1, 2};
    //  std::vector<int64_t> y_crows_data = {0, 1, 3, 5};
    //  const int64_t y_non_zero_num = 5;*/

  /*//  phi::CPUContext dev_ctx_cpu;
  //  dev_ctx_cpu.SetAllocator(
  //      paddle::memory::allocation::AllocatorFacade::Instance()
  //          .GetAllocator(paddle::platform::CPUPlace())
  //          .get());
  //  dev_ctx_cpu.SetHostAllocator(
  //      paddle::memory::allocation::AllocatorFacade::Instance()
  //          .GetAllocator(paddle::platform::CPUPlace())
  //          .get());
  //  dev_ctx_cpu.Init();

  //  phi::CPUContext dev_ctx_cpu;
  //  dev_ctx_cpu.Init();
  //  dev_ctx_cpu.SetAllocator(
  //      paddle::memory::allocation::AllocatorFacade::Instance()
  //          .GetAllocator(phi::CPUPlace())
  //          .get());*/

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(float));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data =
      dense_y.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(float));

  /*//  DenseTensor dense_x =
  //      phi::Empty(alloc.get(),
  //                 DenseTensorMeta(DataType::FLOAT32, {3, 3},
  DataLayout::NCHW));
  //  memcpy(dense_x.data<float>(),
  //         x_dense_data.data(),
  //         x_dense_data.size() * sizeof(float));


  //  DenseTensor dense_y =
  //      phi::Empty(dev_ctx_cpu,
  //                 DenseTensorMeta(DataType::FLOAT32, {3, 3},
  DataLayout::NCHW));
  //  memcpy(dense_y.data<float>(),
  //         y_dense_data.data(),
  //         y_dense_data.size() * sizeof(float));*/

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  auto csr_x = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_y);

  TestElementWiseAddCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  //  TestElementWiseDivideCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

// template <typename T, typename Context>
// void TestElementWiseAddCsrGrad(const Context& dev_ctx_cpu,
//                                const SparseCsrTensor& x,
//                                const SparseCsrTensor& y,
//                                const DDim& dense_dims) {
//   /*  const Context& dev_ctx,
//         const SparseCsrTensor& x,
//         const SparseCsrTensor& y,
//         const SparseCsrTensor& dout,
//         SparseCsrTensor* dx,
//         SparseCsrTensor* dy*/
//   auto out = sparse::ElementWiseAddCsr<T>(dev_ctx_cpu, x, y);
//
//   auto* dx = new SparseCsrTensor();
//   auto* dy = new SparseCsrTensor();
//
//   sparse::CopyCsr(dev_ctx_cpu, y, dev_ctx_cpu.GetPlace(), false, dx);
//   sparse::CopyCsr(dev_ctx_cpu, x, dev_ctx_cpu.GetPlace(), false, dy);
//   sparse::ElementWiseAddCsrGradKernel<T>(dev_ctx_cpu, x, y, out, dx, dy);
//   const DenseTensor expectdx = sparse::SparseCsrToDense<T>(dev_ctx_cpu, x);
//   const DenseTensor expectdy = sparse::SparseCsrToDense<T>(dev_ctx_cpu, y);
//   const DenseTensor densedX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, *dx);
//   const DenseTensor densedY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, *dy);
//   //  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu,
//   //  out);
//
//   //  auto expectdx = x;
//   //  auto expectdy = y;
//
//   for (int j = 0; j < densedX.numel(); ++j) {
//     auto actualResultRow = densedX.template data<T>()[j];
//     auto expectResultRow = expectdx.template data<T>()[j];
//     ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
//   }
// }

TEST(DEV_API, sparse_elementwise_op_csr_grad_kernel) {
  using T = float;
  DDim dense_dims = phi::make_ddim({4, 3});
  std::vector<float> x_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0, 3.2, 0.0, 0.0};

  std::vector<float> y_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5, 0.7, 0.0, 3.5, 0.7};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(float));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data =
      dense_y.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(float));

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

  auto csr_x = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_y);

  auto dx = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_y);
  auto dy = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_x);

  //  TestElementWiseAddCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  /*  const Context& dev_ctx,
        const SparseCsrTensor& x,
        const SparseCsrTensor& y,
        const SparseCsrTensor& dout,
        SparseCsrTensor* dx,
        SparseCsrTensor* dy*/
  auto out = sparse::ElementWiseAddCsr<T>(dev_ctx_cpu, csr_x, csr_y);

  //  auto* dx = new SparseCsrTensor();
  //  auto* dy = new SparseCsrTensor();
  //  sparse::CopyCsr(dev_ctx_cpu, csr_y, dev_ctx_cpu.GetPlace(), false, dx);
  //  sparse::CopyCsr(dev_ctx_cpu, csr_x, dev_ctx_cpu.GetPlace(), false, dy);

  //  sparse::ElementWiseAddCsrGradKernel<T,CPUContext>(
  //      dev_ctx_cpu, csr_x, csr_y, out, &dx, &dy);
  auto dres = sparse::ElementWiseAddCsrGrad<T>(dev_ctx_cpu, csr_x, csr_y, out);
  const DenseTensor expectdx = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor expectdy = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor densedX = sparse::SparseCsrToDense<T>(dev_ctx_cpu, dres[0]);
  const DenseTensor densedY = sparse::SparseCsrToDense<T>(dev_ctx_cpu, dres[1]);
  //  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu,
  //  out);

  //  auto expectdx = x;
  //  auto expectdy = y;

  for (int j = 0; j < densedX.numel(); ++j) {
    auto actualResultRow = densedX.template data<T>()[j];
    auto expectResultRow = expectdx.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}

template <typename T, typename Context>
void TestElementWiseAddGrad(const Context& dev_ctx_cpu,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& y,
                            const DDim& dense_dims) {
  auto out = sparse::ElementWiseAddCsr<T>(dev_ctx_cpu, x, y);
  auto dresult = sparse::ElementWiseAddCsrGrad<T>(dev_ctx_cpu, x, y, out);

  const DenseTensor expectdx = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor expectdy = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor densedX =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);

  for (int j = 0; j < densedX.numel(); ++j) {
    auto actualResultRow = densedX.template data<T>()[j];
    auto expectResultRow = expectdx.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
  for (int j = 0; j < densedY.numel(); ++j) {
    auto actualResultRow = densedY.template data<T>()[j];
    auto expectResultRow = expectdy.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}

template <typename T, typename Context>
void TestElementWiseSubtractGrad(const Context& dev_ctx_cpu,
                                 const SparseCsrTensor& x,
                                 const SparseCsrTensor& y,
                                 const DDim& dense_dims) {
  auto out = sparse::ElementWiseSubtractCsr<T>(dev_ctx_cpu, x, y);
  auto dresult = sparse::ElementWiseSubtractCsrGrad<T>(dev_ctx_cpu, x, y, out);

  const DenseTensor expectdx = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor expectdy = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
  const DenseTensor densedX =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);

  for (int j = 0; j < densedX.numel(); ++j) {
    auto actualResultRow = densedX.template data<T>()[j];
    auto expectResultRow = expectdx.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
  for (int j = 0; j < densedY.numel(); ++j) {
    auto actualResultRow = densedY.template data<T>()[j];
    auto expectResultRow = expectdy.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}

template <typename T, typename Context>
void TestElementWiseMultiplyGrad(const Context& dev_ctx_cpu,
                                 const SparseCsrTensor& x,
                                 const SparseCsrTensor& y,
                                 const DDim& dense_dims) {
  auto out = sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, x, y);
  auto dresult = sparse::ElementWiseMultiplyCsrGrad<T>(dev_ctx_cpu, x, y, out);

  const DenseTensor expectdx = sparse::SparseCsrToDense<T>(
      dev_ctx_cpu, sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, out, y));
  const DenseTensor expectdy = sparse::SparseCsrToDense<T>(
      dev_ctx_cpu, sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, out, x));
  const DenseTensor densedX =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);
  const DenseTensor densedY =
      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);
  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);

  for (int j = 0; j < densedX.numel(); ++j) {
    auto actualResultRow = densedX.template data<T>()[j];
    auto expectResultRow = expectdx.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
  for (int j = 0; j < densedY.numel(); ++j) {
    auto actualResultRow = densedY.template data<T>()[j];
    auto expectResultRow = expectdy.template data<T>()[j];
    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
  }
}

//template <typename T, typename Context>
//void TestElementWiseDivideGrad(const Context& dev_ctx_cpu,
//                                 const SparseCsrTensor& x,
//                                 const SparseCsrTensor& y,
//                                 const DDim& dense_dims) {
//  auto out = sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, x, y);
//  auto dresult = sparse::ElementWiseMultiplyCsrGrad<T>(dev_ctx_cpu, x, y, out);
//
//  const DenseTensor expectdx = sparse::SparseCsrToDense<T>(
//      dev_ctx_cpu, sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, out, y));
//  const DenseTensor expectdy = sparse::SparseCsrToDense<T>(
//      dev_ctx_cpu, sparse::ElementWiseMultiplyCsr<T>(dev_ctx_cpu, out, x));
//  const DenseTensor densedX =
//      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[0]);
//  const DenseTensor densedY =
//      sparse::SparseCsrToDense<T>(dev_ctx_cpu, dresult[1]);
//  const DenseTensor denseOut = sparse::SparseCsrToDense<T>(dev_ctx_cpu, out);
//
//  for (int j = 0; j < densedX.numel(); ++j) {
//    auto actualResultRow = densedX.template data<T>()[j];
//    auto expectResultRow = expectdx.template data<T>()[j];
//    //    std::cout << "actualResultRow: "<< actualResultRow << std::endl;
//    //    std::cout << "expectResultRow: "<< expectResultRow << std::endl;
//    ASSERT_NEAR(expectResultRow, actualResultRow, 1e-6f);
//  }
//}

TEST(DEV_API, sparse_elementwise_op_csr_grad_kernel2) {
  using T = float;
  DDim dense_dims = phi::make_ddim({4, 3});
  std::vector<float> x_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0, 3.2, 0.0, 0.0};

  std::vector<float> y_dense_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5, 0.7, 0.0, 3.5, 0.7};

  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::DenseTensor dense_x(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_x_data =
      dense_x.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_x_data, x_dense_data.data(), x_dense_data.size() * sizeof(float));

  phi::DenseTensor dense_y(
      alloc.get(),
      phi::DenseTensorMeta(DataType::FLOAT32, dense_dims, DataLayout::NCHW));
  auto* dense_y_data =
      dense_y.mutable_data<float>(paddle::platform::CPUPlace());

  memcpy(
      dense_y_data, y_dense_data.data(), y_dense_data.size() * sizeof(float));

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

  auto csr_x = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_x);
  auto csr_y = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_y);

  auto dx = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_y);
  auto dy = sparse::DenseToSparseCsr<float>(dev_ctx_cpu, dense_x);

  //  TestElementWiseAddCsr<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseAddGrad<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseSubtractGrad<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
  TestElementWiseMultiplyGrad<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
//  TestElementWiseDivideGrad<float>(dev_ctx_cpu, csr_x, csr_y, dense_dims);
}

}  // namespace tests
}  // namespace phi
