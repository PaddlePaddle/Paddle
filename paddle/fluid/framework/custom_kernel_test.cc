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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/custom_kernel.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/extension.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_kernel_info_helper.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_factory.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/utils/small_vector.h"

#ifdef _LINUX
// user kernel function
namespace custom_kernel {

// Here we use fake_dot for test
// input 3: two Tensors and one std::vector<Tensor>
// attribute 11: fake_attributes
// output 2: one Tensor* and one std::vector<Tensor*>
template <typename T, typename Context>
void FakeDot(const Context& dev_ctx, const paddle::Tensor& x,
             const paddle::Tensor& y,
             const std::vector<paddle::Tensor>& fake_input_vec,
             bool fake_attr_bool, int fake_attr_int, float fake_attr_float,
             double fake_attr_double, int64_t fake_attr_int64,
             pten::dtype::float16 fake_attr_f16, pten::DataType fake_attr_dtype,
             const pten::Scalar& fake_attr_scalar,
             const pten::ScalarArray& fake_attr_scalar_array,
             const std::vector<int64_t>& fake_attr_int64_vec,
             const std::vector<int>& fake_attr_int_vec, paddle::Tensor* out,
             std::vector<paddle::Tensor*> fake_out_vec) {
  // print param info
  std::cout << "fake_input_vec.size: " << fake_input_vec.size() << std::endl;
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
  std::cout << "fake_out_vec: " << fake_out_vec.size() << std::endl;

  // assert check
  assert(fake_input_vec.size() == 2);
  assert(fake_attr_bool == false);
  assert(fake_attr_int == 1);
  assert(fake_attr_float == 2);
  assert(fake_attr_double == 3);
  assert(fake_attr_int64 == 4);
  assert(fake_attr_f16 == pten::dtype::float16(5));
  assert(fake_attr_dtype == pten::DataType::UINT32);
  assert(fake_attr_int64_vec.size() == 0);
  assert(fake_attr_int_vec.size() == 0);
  assert(fake_out_vec.size() == 2);

  auto const *x_ptr = x.data<T>(), *x_ptr_ = &x_ptr[0];
  auto const *y_ptr = y.data<T>(), *y_ptr_ = &y_ptr[0];
  auto* z = out->mutable_data<T>(paddle::PlaceType::kCPU);
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

PD_REGISTER_KERNEL(fake_dot, CPU, ALL_LAYOUT, custom_kernel::FakeDot, float,
                   double, int, int64_t, int8_t, uint8_t) {}

// Upper code will store dot kernels info into OpKernelInfoMap
TEST(CustomKernel, custom_kernel_dot) {
  std::string op_name = "fake_dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ALL_LAYOUT;

  // 1.custom kernel info parsed and store
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance().GetMap().find(op_name) !=
              paddle::OpKernelInfoMap::Instance().GetMap().end());

  // 2.info check
  EXPECT_EQ(
      6, static_cast<int>(paddle::OpKernelInfoMap::Instance()[op_name].size()));
  // index 0
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][0].GetBackend() ==
              backend);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][0].GetDataLayout() ==
              layout);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][0].GetDataType() ==
              pten::DataType::FLOAT32);
  // index 5
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][5].GetBackend() ==
              backend);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][5].GetDataLayout() ==
              layout);
  EXPECT_TRUE(paddle::OpKernelInfoMap::Instance()[op_name][5].GetDataType() ==
              pten::DataType::UINT8);

  // 3.before register
  auto& kernel_factory_instance = pten::KernelFactory::Instance();
  auto& kernels = pten::KernelFactory::Instance().kernels();
  EXPECT_TRUE(!kernel_factory_instance.HasCompatiblePtenKernel(op_name));

  // mock fake_dot is supported by pten for HasCompatiblePtenKernel check while
  // registering
  auto& fake_dot_kernels = kernels[op_name];

  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::FLOAT32)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::FLOAT64)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT32)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT64)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT8)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::UINT8)) ==
              fake_dot_kernels.end());

  // register
  paddle::framework::RegisterKernelWithMetaInfoMap(
      paddle::OpKernelInfoMap::Instance());

  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::FLOAT32)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::FLOAT64)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT32)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT64)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::INT8)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  pten::KernelKey(backend, layout, pten::DataType::UINT8)) !=
              fake_dot_kernels.end());

  // 4.kernel select
  auto kernel = kernel_factory_instance.SelectKernelOrThrowError(
      op_name, pten::KernelKey(backend, layout, pten::DataType::UINT8));

  // 5.prepare parameters for kernel
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::UINT8,
                                         pten::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<uint8_t>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<pten::DenseTensor>(
      alloc.get(), pten::DenseTensorMeta(pten::DataType::UINT8,
                                         pten::framework::make_ddim({2, 3}),
                                         pten::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<uint8_t>(paddle::platform::CPUPlace());

  // dot x,y and result
  uint8_t sum[2] = {0, 0};
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dense_x_data[i * 3 + j] = (i * 3 + j);
      dense_y_data[i * 3 + j] = (i * 3 + j);
      sum[i] += (i * 3 + j) * (i * 3 + j);
    }
  }

  // 6.prepare kernel_context
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  auto kernel_context = pten::KernelContext(dev_ctx);
  kernel_context.EmplaceBackInput(dense_x.get());  // idx:0, index:[0,1)
  kernel_context.EmplaceBackInput(dense_y.get());  // idx:1, index:[1,2)

  // fake_input_vec: idx:2, index:[2,4)
  size_t fake_input_vec_idx = 2;
  size_t fake_input_vec_index_start = 2;
  size_t fake_input_vec_index_end = 4;
  kernel_context.EmplaceBackInputWithoutSetRange(dense_x.get());
  kernel_context.EmplaceBackInputWithoutSetRange(dense_y.get());
  kernel_context.AssignInputRange(
      std::make_pair(fake_input_vec_index_start, fake_input_vec_index_end),
      fake_input_vec_idx);

  bool fake_attr_bool = false;
  int fake_attr_int = 1;
  float fake_attr_float = 2.0;
  double fake_attr_double = 3.0;
  int64_t fake_attr_int64 = 4;
  pten::dtype::float16 fake_attr_f16 = pten::dtype::float16(5);
  pten::DataType fake_attr_dtype = pten::DataType::UINT32;
  paddle::framework::LoDTensor tmp_tensor;
  tmp_tensor.mutable_data<uint8_t>({1}, pten::TransToPtenPlace(backend));
  pten::Scalar fake_attr_scalar{tmp_tensor};
  pten::ScalarArray fake_attr_scalar_array;
  std::vector<int64_t> fake_attr_int64_vec;
  std::vector<int> fake_attr_int_vec;

  kernel_context.EmplaceBackAttr(fake_attr_bool);
  kernel_context.EmplaceBackAttr(fake_attr_int);
  kernel_context.EmplaceBackAttr(fake_attr_float);
  kernel_context.EmplaceBackAttr(fake_attr_double);
  kernel_context.EmplaceBackAttr(fake_attr_int64);
  kernel_context.EmplaceBackAttr(fake_attr_f16);
  kernel_context.EmplaceBackAttr(fake_attr_dtype);
  kernel_context.EmplaceBackAttr(fake_attr_scalar);
  kernel_context.EmplaceBackAttr(fake_attr_scalar_array);
  kernel_context.EmplaceBackAttr(fake_attr_int64_vec);
  kernel_context.EmplaceBackAttr(fake_attr_int_vec);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToPtenPlace(backend)),
      pten::DenseTensorMeta());

  pten::MetaTensor meta_out(dense_out.get());
  pten::DotInferMeta(*dense_x, *dense_y, &meta_out);
  kernel_context.EmplaceBackOutput(dense_out.get());  // idx:0 index:[0,1)

  // fake_input_vec: idx:1, index:[1,3)
  size_t fake_out_vec_idx = 1;
  size_t fake_out_vec_index_start = 1;
  size_t fake_out_vec_index_end = 3;
  kernel_context.EmplaceBackOutputWithoutSetRange(dense_out.get());
  kernel_context.EmplaceBackOutputWithoutSetRange(dense_out.get());
  kernel_context.AssignOutputRange(
      std::make_pair(fake_out_vec_index_start, fake_out_vec_index_end),
      fake_out_vec_idx);

  // 7.kernel call
  kernel(&kernel_context);

  // 8.check result
  ASSERT_EQ(dense_out->dims().size(), 2);
  ASSERT_EQ(dense_out->dims()[0], 2);
  ASSERT_EQ(dense_out->numel(), 2);
  ASSERT_EQ(dense_out->dtype(), pten::DataType::UINT8);
  ASSERT_EQ(dense_out->layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(dense_out->initialized(), true);

  auto expect_result = sum;
  auto actual_result0 = dense_out->data<uint8_t>()[0];
  auto actual_result1 = dense_out->data<uint8_t>()[1];
  ASSERT_EQ(expect_result[0], actual_result0);
  ASSERT_EQ(expect_result[1], actual_result1);
}

// test OpKernelInfoHelper
TEST(OpKernelInfoHelper, op_kernel_info_help_getters) {
  using OpKernelInfoHelper = paddle::framework::OpKernelInfoHelper;
  std::string op_name = "fake_dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  auto op_kernel_info = paddle::OpKernelInfoMap::Instance()[op_name][0];

  EXPECT_EQ(op_name, OpKernelInfoHelper::GetOpName(op_kernel_info));
  EXPECT_EQ(backend, OpKernelInfoHelper::GetBackend(op_kernel_info));
  EXPECT_EQ(layout, OpKernelInfoHelper::GetDataLayout(op_kernel_info));
  EXPECT_EQ(dtype, OpKernelInfoHelper::GetDataType(op_kernel_info));

  EXPECT_EQ(pten::KernelKey(backend, layout, dtype),
            OpKernelInfoHelper::GetKernelKey(op_kernel_info));

  paddle::CustomKernelFunc kernel_fn =
      PD_PT_KERNEL(custom_kernel::FakeDot<float, paddle::CPUContext>);
  EXPECT_EQ(kernel_fn, OpKernelInfoHelper::GetKernelFn(op_kernel_info));

  void* variadic_func =
      PD_PT_VARIADIC_KERNEL(custom_kernel::FakeDot<float, paddle::CPUContext>);
  EXPECT_EQ(variadic_func,
            OpKernelInfoHelper::GetVariadicKernelFn(op_kernel_info));

  auto& input_defs = OpKernelInfoHelper::GetInputDefs(op_kernel_info);
  auto& output_defs = OpKernelInfoHelper::GetOutputDefs(op_kernel_info);
  auto& attribute_defs = OpKernelInfoHelper::GetAttributeDefs(op_kernel_info);
  EXPECT_EQ(3, static_cast<int>(input_defs.size()));
  EXPECT_EQ(2, static_cast<int>(output_defs.size()));
  EXPECT_EQ(11, static_cast<int>(attribute_defs.size()));
}
#endif
