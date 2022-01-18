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

#include "paddle/fluid/framework/custom_kernel.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/extension.h"
#include "paddle/fluid/framework/op_kernel_info_helper.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_factory.h"
#include "paddle/pten/infermeta/binary.h"

// user kernel function
namespace custom_kernel {

template <typename T>
void Dot(const paddle::CPUContext& dev_ctx, const paddle::Tensor& x,
         const paddle::Tensor& y, paddle::Tensor* out) {
  auto const *x_ptr = x.data<T>(), *x_ptr_ = &x_ptr[0];
  auto const *y_ptr = y.data<T>(), *y_ptr_ = &y_ptr[0];
  auto* z = out->mutable_data<T>();
  auto shape = x.shape();
  auto const N = x.numel();
  auto const B = shape[shape.size() - 1];
  for (int j = 0; j < N / B; j++) {
    T ss = 0;
    for (int i = 0; i < B; i++) ss += (*x_ptr_++) * (*y_ptr_++);
    z[j] = ss;
  }
}
}  // namespace custom_kernel

// register
PD_REGISTER_KERNEL(dot, CPU, ANY, INT8, custom_kernel::Dot<int8_t>) {
  /* do some args define here
   * the only param can be used is OpKernelInfo* kernel */
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

TEST(CustomKernel, custom_kernel_register_and_run) {
  // kernel info
  // dot <cpu, any, int8> is not implemented
  std::string op_name = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::INT8;

  // 1.custom kernel info parsed and store
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance().GetMap().find("dot") !=
              paddle::OpKernelInfoMap::Instance().GetMap().end());

  // 2.info check
  EXPECT_EQ(1, paddle::OpKernelInfoMap::Instance()["dot"].size());
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetBackend() ==
              backend);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetDataLayout() ==
              layout);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetDataType() ==
              dtype);

  // 3.register
  EXPECT_TRUE(pten::KernelFactory::Instance().kernels().end() !=
              pten::KernelFactory::Instance().kernels().find("dot"));
  pten::KernelKey kernel_key(backend, layout, dtype);
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key) ==
      pten::KernelFactory::Instance().kernels()["dot"].end());

  paddle::framework::RegisterKernelWithMetaInfoMap(
      paddle::OpKernelInfoMap::Instance());

  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key) !=
      pten::KernelFactory::Instance().kernels()["dot"].end());

  // 4.prepare input
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::INT8,
                                         paddle::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<int8_t>();

  auto dense_y = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::INT8,
                                         paddle::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_y_data = dense_y->mutable_data<int8_t>();

  int8_t sum[2] = {0, 0};
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dense_x_data[i * 3 + j] = (i * 3 + j);
      dense_y_data[i * 3 + j] = (i * 3 + j);
      sum[i] += (i * 3 + j) * (i * 3 + j);
    }
  }
  // 5.kernel select
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      op_name, kernel_key);

  // 6.prepare kernel_context
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  auto kernel_context = pten::KernelContext(dev_ctx);
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackInput(dense_y);

  auto out_meta = pten::DotInferMeta(dense_x->meta(), dense_x->meta());
  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(backend)),
      std::move(out_meta));
  kernel_context.EmplaceBackOutput(dense_out);

  // 7.kernel call
  kernel(&kernel_context);

  // 8.check result
  ASSERT_EQ(dense_out->dims().size(), 2);
  ASSERT_EQ(dense_out->dims()[0], 2);
  ASSERT_EQ(dense_out->numel(), 2);
  ASSERT_EQ(dense_out->dtype(), pten::DataType::INT8);
  ASSERT_EQ(dense_out->layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(dense_out->initialized(), true);

  auto expect_result = sum;
  auto actual_result0 = dense_out->data<int8_t>()[0];
  auto actual_result1 = dense_out->data<int8_t>()[1];
  ASSERT_EQ(expect_result[0], actual_result0);
  ASSERT_EQ(expect_result[1], actual_result1);
}

// test OpKernelInfoHelper
TEST(OpKernelInfoHelper, GetOpName) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::INT8;

  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);

  EXPECT_TRUE(op_name_1 == paddle::framework::OpKernelInfoHelper::GetOpName(
                               op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetKernelKey) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);

  EXPECT_TRUE(
      pten::KernelKey(backend, layout, dtype) ==
      paddle::framework::OpKernelInfoHelper::GetKernelKey(op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetKernelFn) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  paddle::CustomKernelFunc kernel_fn{nullptr};
  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);
  op_kernel_info_1.SetKernelFn(std::move(kernel_fn));

  EXPECT_TRUE(kernel_fn == paddle::framework::OpKernelInfoHelper::GetKernelFn(
                               op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetVariadicKernelFn) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  void* variadic_func{nullptr};
  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);
  op_kernel_info_1.SetVariadicKernelFn(std::move(variadic_func));

  EXPECT_TRUE(variadic_func ==
              paddle::framework::OpKernelInfoHelper::GetVariadicKernelFn(
                  op_kernel_info_1));
}
