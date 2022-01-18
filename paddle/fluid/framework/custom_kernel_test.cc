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
#include "paddle/utils/small_vector.h"

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

template <typename T>
void FakeDot(const paddle::CPUContext& dev_ctx, const paddle::Tensor& x,
             const paddle::Tensor& y, bool fake_attr_bool, int fake_attr_int,
             float fake_attr_float, double fake_attr_double,
             int64_t fake_attr_int64, paddle::platform::float16 fake_attr_f16,
             pten::DataType fake_attr_dtype,
             const pten::ScalarArray& fake_attr_scalar_array,
             const std::vector<int64_t>& fake_attr_int64_vec,
             const std::vector<int>& fake_attr_int_vec, paddle::Tensor* out) {
  // print param info
  std::cout << "fake_attr_bool: " << fake_attr_bool << std::endl;
  std::cout << "fake_attr_int: " << fake_attr_int << std::endl;
  std::cout << "fake_attr_float: " << fake_attr_float << std::endl;
  std::cout << "fake_attr_double: " << fake_attr_double << std::endl;
  std::cout << "fake_attr_int64: " << fake_attr_int64 << std::endl;
  std::cout << "fake_attr_f16: " << fake_attr_f16 << std::endl;
  std::cout << "fake_attr_dtype: " << fake_attr_dtype << std::endl;
  std::cout << "fake_attr_int64_vec: " << fake_attr_int64_vec.size()
            << std::endl;
  std::cout << "fake_attr_int_vec: " << fake_attr_int_vec.size() << std::endl;

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

// register
PD_REGISTER_KERNEL(dot, CPU, ANY, UINT8, custom_kernel::FakeDot<uint8_t>) {
  /* do some args define here
   * the only param can be used is OpKernelInfo* kernel */
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

// Upper code will store dot kernels info into OpKernelInfoMap
// Here we use dot <CPU, ANY, INT8> and <CPU, ANY, UINT8>
// This test will fail when these two kernels are aupported in framework
TEST(CustomKernel, custom_kernel_dot) {
  std::string op_name = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype_int8 = pten::DataType::INT8;
  pten::DataType dtype_uint8 = pten::DataType::UINT8;

  // 1.custom kernel info parsed and store
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance().GetMap().find("dot") !=
              paddle::OpKernelInfoMap::Instance().GetMap().end());

  // 2.info check
  EXPECT_EQ(
      2, static_cast<int>(paddle::OpKernelInfoMap::Instance()["dot"].size()));
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetBackend() ==
              backend);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetDataLayout() ==
              layout);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][0].GetDataType() ==
              dtype_int8);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][1].GetBackend() ==
              backend);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][1].GetDataLayout() ==
              layout);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()["dot"][1].GetDataType() ==
              dtype_uint8);

  // 3.register
  EXPECT_TRUE(pten::KernelFactory::Instance().kernels().end() !=
              pten::KernelFactory::Instance().kernels().find("dot"));

  pten::KernelKey kernel_key_int8(backend, layout, dtype_int8);
  pten::KernelKey kernel_key_uint8(backend, layout, dtype_uint8);
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key_int8) ==
      pten::KernelFactory::Instance().kernels()["dot"].end());
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key_uint8) ==
      pten::KernelFactory::Instance().kernels()["dot"].end());

  paddle::framework::RegisterKernelWithMetaInfoMap(
      paddle::OpKernelInfoMap::Instance());

  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key_int8) !=
      pten::KernelFactory::Instance().kernels()["dot"].end());
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key_uint8) !=
      pten::KernelFactory::Instance().kernels()["dot"].end());

  // 4.prepare input for int8 kernel
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
      op_name, kernel_key_int8);

  // 6.prepare kernel_context
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  auto kernel_context = pten::KernelContext(dev_ctx);
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackInput(dense_y);

  auto out_meta = pten::DotInferMeta(dense_x->meta(), dense_y->meta());
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

  // fake_dot check
  auto dense_x_1 = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::UINT8,
                                         paddle::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_x_1_data = dense_x_1->mutable_data<uint8_t>();

  auto dense_y_1 = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::UINT8,
                                         paddle::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_y_1_data = dense_y_1->mutable_data<uint8_t>();

  uint8_t sum_1[2] = {0, 0};
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dense_x_1_data[i * 3 + j] = (i * 3 + j);
      dense_y_1_data[i * 3 + j] = (i * 3 + j);
      sum_1[i] += (i * 3 + j) * (i * 3 + j);
    }
  }

  // 5.kernel select
  auto kernel_1 = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      op_name, kernel_key_uint8);
  // 6.prepare kernel_context
  auto kernel_context_1 = pten::KernelContext(dev_ctx);
  kernel_context_1.EmplaceBackInput(dense_x_1);
  kernel_context_1.EmplaceBackInput(dense_y_1);

  bool fake_attr_bool = false;
  int fake_attr_int = 1;
  float fake_attr_float = 2.0;
  double fake_attr_double = 3.0;
  int64_t fake_attr_int64 = 4;
  paddle::platform::float16 fake_attr_f16 = paddle::platform::float16(5);
  pten::DataType fake_attr_dtype = pten::DataType::INT32;
  pten::ScalarArray fake_attr_scalar_array;
  std::vector<int64_t> fake_attr_int64_vec;
  std::vector<int> fake_attr_int_vec;

  kernel_context_1.EmplaceBackAttr(fake_attr_bool);
  kernel_context_1.EmplaceBackAttr(fake_attr_int);
  kernel_context_1.EmplaceBackAttr(fake_attr_float);
  kernel_context_1.EmplaceBackAttr(fake_attr_double);
  kernel_context_1.EmplaceBackAttr(fake_attr_int64);
  kernel_context_1.EmplaceBackAttr(fake_attr_f16);
  kernel_context_1.EmplaceBackAttr(fake_attr_dtype);
  kernel_context_1.EmplaceBackAttr(fake_attr_scalar_array);
  kernel_context_1.EmplaceBackAttr(fake_attr_int64_vec);
  kernel_context_1.EmplaceBackAttr(fake_attr_int_vec);

  auto out_meta_1 = pten::DotInferMeta(dense_x_1->meta(), dense_y_1->meta());
  auto dense_out_1 = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(backend)),
      std::move(out_meta_1));
  kernel_context_1.EmplaceBackOutput(dense_out_1);

  // 7.kernel call
  kernel_1(&kernel_context_1);

  // 8.check result
  ASSERT_EQ(dense_out_1->dims().size(), 2);
  ASSERT_EQ(dense_out_1->dims()[0], 2);
  ASSERT_EQ(dense_out_1->numel(), 2);
  ASSERT_EQ(dense_out_1->dtype(), pten::DataType::UINT8);
  ASSERT_EQ(dense_out_1->layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(dense_out_1->initialized(), true);

  auto expect_result_1 = sum_1;
  auto actual_result0_1 = dense_out_1->data<uint8_t>()[0];
  auto actual_result1_1 = dense_out_1->data<uint8_t>()[1];
  ASSERT_EQ(expect_result_1[0], actual_result0_1);
  ASSERT_EQ(expect_result_1[1], actual_result1_1);
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
