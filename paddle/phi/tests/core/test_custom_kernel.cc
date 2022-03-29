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

#include <gtest/gtest.h>

#ifdef _LINUX
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"

// user kernel function
namespace custom_kernel {

// Here we use fake_dot for test
// input 3: two Tensors and one std::vector<Tensor>
// attribute 11: fake_attributes
// output 2: one Tensor* and one std::vector<Tensor*>
template <typename T, typename Context>
void FakeDot(const Context& dev_ctx,
             const phi::DenseTensor& x,
             const phi::DenseTensor& y,
             const std::vector<const phi::DenseTensor*>& fake_input_vec,
             bool fake_attr_bool,
             int fake_attr_int,
             float fake_attr_float,
             double fake_attr_double,
             int64_t fake_attr_int64,
             phi::dtype::float16 fake_attr_f16,
             phi::DataType fake_attr_dtype,
             const phi::Scalar& fake_attr_scalar,
             const phi::ScalarArray& fake_attr_scalar_array,
             const std::vector<int64_t>& fake_attr_int64_vec,
             const std::vector<int>& fake_attr_int_vec,
             phi::DenseTensor* out,
             std::vector<phi::DenseTensor*> fake_out_vec) {
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
  assert(fake_attr_f16 == phi::dtype::float16(5));
  assert(fake_attr_dtype == phi::DataType::UINT32);
  assert(fake_attr_int64_vec.size() == 0);
  assert(fake_attr_int_vec.size() == 0);
  assert(fake_out_vec.size() == 2);

  auto const *x_ptr = x.data<T>(), *x_ptr_ = &x_ptr[0];
  auto const *y_ptr = y.data<T>(), *y_ptr_ = &y_ptr[0];
  T* z = dev_ctx.template Alloc<T>(out);
  auto&& d = x.dims();
  auto const N = x.numel();
  auto const B = d[d.size() - 1];
  for (int j = 0; j < N / B; j++) {
    T ss = 0;
    for (int i = 0; i < B; i++) ss += (*x_ptr_++) * (*y_ptr_++);
    z[j] = ss;
  }
}
}  // namespace custom_kernel

PD_REGISTER_BUILTIN_KERNEL(fake_dot,
                           CPU,
                           ALL_LAYOUT,
                           custom_kernel::FakeDot,
                           float,
                           double,
                           int,
                           int64_t,
                           int8_t,
                           uint8_t) {}

namespace phi {
namespace tests {

// Upper code will store dot kernels info into OpKernelInfoMap
TEST(CustomKernel, custom_kernel_dot) {
  std::string op_name = "fake_dot";
  phi::Backend backend = phi::Backend::CPU;
  phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT;

  // 1.custom kernel info parsed and store
  EXPECT_TRUE(phi::CustomKernelMap::Instance().GetMap().find(op_name) !=
              phi::CustomKernelMap::Instance().GetMap().end());

  auto& custom_kernels = phi::CustomKernelMap::Instance().Kernels();
  // 2.info check
  EXPECT_EQ(6, static_cast<int>(custom_kernels[op_name].size()));
  auto& custom_fake_dot_kernels = custom_kernels[op_name];
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT32)) !=
              custom_fake_dot_kernels.end());
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT64)) !=
              custom_fake_dot_kernels.end());
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT32)) !=
              custom_fake_dot_kernels.end());
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT64)) !=
              custom_fake_dot_kernels.end());
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT8)) !=
              custom_fake_dot_kernels.end());
  EXPECT_TRUE(custom_fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::UINT8)) !=
              custom_fake_dot_kernels.end());

  // 3.before register
  auto& kernels = phi::KernelFactory::Instance().kernels();
  EXPECT_TRUE(kernels.find(op_name) == kernels.end());

  // mock fake_dot is supported by phi for check while registering
  auto& fake_dot_kernels = kernels[op_name];

  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT32)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT64)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT32)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT64)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT8)) ==
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::UINT8)) ==
              fake_dot_kernels.end());

  // register
  phi::CustomKernelMap::Instance().RegisterCustomKernels();

  EXPECT_EQ(0, static_cast<int>(custom_fake_dot_kernels.size()));

  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT32)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::FLOAT64)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT32)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT64)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::INT8)) !=
              fake_dot_kernels.end());
  EXPECT_TRUE(fake_dot_kernels.find(
                  phi::KernelKey(backend, layout, phi::DataType::UINT8)) !=
              fake_dot_kernels.end());

  // 4.kernel select
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      op_name, phi::KernelKey(backend, layout, phi::DataType::UINT8));

  // 5.prepare parameters for kernel
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::UINT8, phi::make_ddim({2, 3}), phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<uint8_t>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::UINT8, phi::make_ddim({2, 3}), phi::DataLayout::NCHW));
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
  auto kernel_context = phi::KernelContext(dev_ctx);
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
  phi::dtype::float16 fake_attr_f16 = phi::dtype::float16(5);
  phi::DataType fake_attr_dtype = phi::DataType::UINT32;
  paddle::framework::LoDTensor tmp_tensor;
  tmp_tensor.mutable_data<uint8_t>({1}, phi::TransToPhiPlace(backend));
  phi::Scalar fake_attr_scalar{tmp_tensor};
  phi::ScalarArray fake_attr_scalar_array;
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

  auto dense_out = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPhiPlace(backend)),
      phi::DenseTensorMeta());

  phi::MetaTensor meta_out(dense_out.get());
  phi::DotInferMeta(*dense_x, *dense_y, &meta_out);
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
  ASSERT_EQ(dense_out->dtype(), phi::DataType::UINT8);
  ASSERT_EQ(dense_out->layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(dense_out->initialized(), true);

  auto expect_result = sum;
  auto actual_result0 = dense_out->data<uint8_t>()[0];
  auto actual_result1 = dense_out->data<uint8_t>()[1];
  ASSERT_EQ(expect_result[0], actual_result0);
  ASSERT_EQ(expect_result[1], actual_result1);
}

}  // namespace tests
}  // namespace phi

#endif
