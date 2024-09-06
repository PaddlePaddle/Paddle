// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/selected_rows.h"

PD_DECLARE_KERNEL(empty, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(empty, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace tests {

using Tensor = paddle::Tensor;
using DataType = phi::DataType;

template <typename T>
Tensor InitCPUTensorForTest() {
  std::vector<int64_t> tensor_shape{5, 5};
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  Tensor t1 = paddle::experimental::empty(tensor_shape, dtype, phi::CPUPlace());
  auto* p_data_ptr = t1.data<T>();
  for (int64_t i = 0; i < t1.size(); i++) {
    p_data_ptr[i] = T(5);
  }
  return t1;
}

template <typename T>
void TestCopyTensor() {
  auto t1 = InitCPUTensorForTest<T>();
  auto t1_cpu_cp = t1.copy_to(phi::CPUPlace(), /*blocking=*/false);
  PADDLE_ENFORCE_EQ(t1_cpu_cp.place(),
                    phi::CPUPlace(),
                    common::errors::InvalidArgument("t1_cpu_cp should copy to "
                                                    "CPUPlace, but got %s",
                                                    t1_cpu_cp.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    PADDLE_ENFORCE_EQ(
        t1_cpu_cp.template data<T>()[i],
        T(5),
        common::errors::InvalidArgument(
            "t1_cpu_cp.template data<T>()[%d] should be equal to T(5) ", i));
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  VLOG(2) << "Do GPU copy test";
  auto t1_gpu_cp = t1_cpu_cp.copy_to(phi::GPUPlace(), /*blocking=*/false);
  PADDLE_ENFORCE_EQ(t1_gpu_cp.place(),
                    phi::GPUPlace(),
                    common::errors::InvalidArgument("t1_gpu_cp should copy to "
                                                    "GPUPlace, but got %s",
                                                    t1_gpu_cp.place()));
  auto t1_gpu_cp_cp = t1_gpu_cp.copy_to(phi::GPUPlace(), /*blocking=*/false);
  PADDLE_ENFORCE_EQ(
      t1_gpu_cp_cp.place(),
      phi::GPUPlace(),
      common::errors::InvalidArgument("t1_gpu_cp_cp should copy to "
                                      "GPUPlace, but got %s",
                                      t1_gpu_cp_cp.place()));
  auto t1_gpu_cp_cp_cpu =
      t1_gpu_cp_cp.copy_to(phi::CPUPlace(), /*blocking=*/false);
  PADDLE_ENFORCE_EQ(
      t1_gpu_cp_cp_cpu.place(),
      phi::CPUPlace(),
      common::errors::InvalidArgument("t1_gpu_cp_cp_cpu should copy to "
                                      "CPUPlace, but got %s",
                                      t1_gpu_cp_cp_cpu.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    PADDLE_ENFORCE_EQ(
        t1_gpu_cp_cp_cpu.template data<T>()[i],
        T(5),
        common::errors::InvalidArgument(
            "t1_gpu_cp_cp_cpu.template data<T>()[%d] should be equal to T(5) ",
            i));
  }
#endif
}

void TestAPIPlace() {
  std::vector<int64_t> tensor_shape = {5, 5};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto t1 = paddle::experimental::empty(
      tensor_shape, DataType::FLOAT32, phi::GPUPlace());
  PADDLE_ENFORCE_EQ(t1.place(),
                    phi::GPUPlace(),
                    common::errors::InvalidArgument(
                        "t1 should copy to GPUPlace, but got %s", t1.place()));
#endif
  auto t2 = paddle::experimental::empty(
      tensor_shape, DataType::FLOAT32, phi::CPUPlace());
  PADDLE_ENFORCE_EQ(t2.place(),
                    phi::CPUPlace(),
                    common::errors::InvalidArgument(
                        "t2 should copy to CPUPlace, but got %s", t2.place()));
}

void TestAPISizeAndShape() {
  std::vector<int64_t> tensor_shape = {5, 5};
  auto t1 = paddle::experimental::empty(tensor_shape);
  PADDLE_ENFORCE_EQ(
      t1.size(),
      25,
      common::errors::InvalidArgument("t1.size should be equal to 25, "
                                      "but got %d",
                                      t1.size()));
  PADDLE_ENFORCE_EQ(t1.shape(),
                    tensor_shape,
                    common::errors::InvalidArgument(
                        "t1.shape should be equal to tensor_shape, "));
}

void TestAPISlice() {
  std::vector<int64_t> tensor_shape_origin1 = {5, 5};
  std::vector<int64_t> tensor_shape_sub1 = {3, 5};
  std::vector<int64_t> tensor_shape_origin2 = {5, 5, 5};
  std::vector<int64_t> tensor_shape_sub2 = {1, 5, 5};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto t1 = paddle::experimental::empty(
      tensor_shape_origin1, DataType::FLOAT32, phi::GPUPlace());
  PADDLE_ENFORCE_EQ(
      t1.slice(0, 5).shape(),
      tensor_shape_origin1,
      common::errors::InvalidArgument("t1.slice(0, 5).shape should be equal to "
                                      "{5, 5}"));
  PADDLE_ENFORCE_EQ(
      t1.slice(0, 3).shape(),
      tensor_shape_sub1,
      common::errors::InvalidArgument("t1.slice(0, 3).shape should be equal to "
                                      "{3, 5}"));
  auto t2 = paddle::experimental::empty(
      tensor_shape_origin2, DataType::FLOAT32, phi::GPUPlace());
  PADDLE_ENFORCE_EQ(
      t2.slice(4, 5).shape(),
      tensor_shape_sub2,
      common::errors::InvalidArgument("t2.slice(4, 5).shape should be equal to "
                                      "{1, 5, 5}"));
#endif
  auto t3 = paddle::experimental::empty(
      tensor_shape_origin1, DataType::FLOAT32, phi::CPUPlace());
  PADDLE_ENFORCE_EQ(
      t3.slice(0, 5).shape(),
      tensor_shape_origin1,
      common::errors::InvalidArgument("t3.slice(0, 5).shape should be equal to "
                                      "{5, 5}"));
  PADDLE_ENFORCE_EQ(
      t3.slice(0, 3).shape(),
      tensor_shape_sub1,
      common::errors::InvalidArgument("t3.slice(0, 3).shape should be equal to "
                                      "{3, 5}"));
  auto t4 = paddle::experimental::empty(
      tensor_shape_origin2, DataType::FLOAT32, phi::CPUPlace());
  PADDLE_ENFORCE_EQ(
      t4.slice(4, 5).shape(),
      tensor_shape_sub2,
      common::errors::InvalidArgument("t4.slice(4, 5).shape should be equal to "
                                      "{1, 5, 5}"));

  // Test writing function for sliced tensor
  auto t = InitCPUTensorForTest<float>();
  auto t_sliced = t.slice(0, 1);
  auto* t_sliced_data_ptr = t_sliced.data<float>();
  for (int64_t i = 0; i < t_sliced.size(); i++) {
    t_sliced_data_ptr[i] += static_cast<float>(5);
  }
  auto* t_data_ptr = t.data<float>();
  for (int64_t i = 0; i < t_sliced.size(); i++) {
    PADDLE_ENFORCE_EQ(t_data_ptr[i],
                      static_cast<float>(10),
                      common::errors::InvalidArgument(
                          "Required t_data_ptr[%d] should be equal "
                          "to static_cast<float>(10) ",
                          i));
  }
}

template <typename T>
paddle::DataType TestDtype() {
  std::vector<int64_t> tensor_shape = {5, 5};
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  auto t1 = paddle::experimental::empty(tensor_shape, dtype, phi::CPUPlace());
  return t1.type();
}

template <typename T>
void TestCast(paddle::DataType data_type) {
  std::vector<int64_t> tensor_shape = {5, 5};
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  auto t1 = paddle::experimental::empty(tensor_shape, dtype, phi::CPUPlace());
  auto t2 = t1.cast(data_type);
  PADDLE_ENFORCE_EQ(
      t2.type(),
      data_type,
      common::errors::InvalidArgument("t2.type() should be equal to data_type, "
                                      "but got %s",
                                      t2.type()));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto tg1 = paddle::experimental::empty(tensor_shape, dtype, phi::GPUPlace());
  auto tg2 = tg1.cast(data_type);
  PADDLE_ENFORCE_EQ(tg2.type(),
                    data_type,
                    common::errors::InvalidArgument(
                        "tg2.type() should be equal to data_type, "
                        "but got %s",
                        tg2.type()));
#endif
}

void GroupTestCopy() {
  VLOG(2) << "Float cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<float>();
  VLOG(2) << "Double cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<double>();
  VLOG(2) << "int cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int32_t>();
  VLOG(2) << "int64 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int64_t>();
  VLOG(2) << "int16 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int16_t>();
  VLOG(2) << "int8 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int8_t>();
  VLOG(2) << "uint8 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<uint8_t>();
  VLOG(2) << "complex<float> cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::complex64>();
  VLOG(2) << "complex<double> cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::complex128>();
  VLOG(2) << "Fp16 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::float16>();
}

void GroupTestCast() {
  VLOG(2) << "int16_t cast";
  TestCast<int16_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "int32 cast";
  TestCast<int32_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "int64 cast";
  TestCast<int64_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "double cast";
  TestCast<double>(paddle::DataType::FLOAT32);
  VLOG(2) << "bool cast";
  TestCast<bool>(paddle::DataType::FLOAT32);
  VLOG(2) << "uint8 cast";
  TestCast<uint8_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "float cast";
  TestCast<float>(paddle::DataType::FLOAT32);
  VLOG(2) << "complex<float> cast";
  TestCast<paddle::complex64>(paddle::DataType::FLOAT32);
  VLOG(2) << "complex<double> cast";
  TestCast<paddle::complex128>(paddle::DataType::FLOAT32);
  VLOG(2) << "float16 cast";
  TestCast<paddle::float16>(paddle::DataType::FLOAT16);
}

void GroupTestDtype() {
  PADDLE_ENFORCE_EQ(
      TestDtype<bool>(),
      paddle::DataType::BOOL,
      common::errors::InvalidArgument("TestDtype<bool>() should be equal to "
                                      "paddle::DataType::BOOL, but got %s",
                                      TestDtype<bool>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<int8_t>(),
      paddle::DataType::INT8,
      common::errors::InvalidArgument("TestDtype<int8_t>() should be equal to "
                                      "paddle::DataType::INT8, but got %s",
                                      TestDtype<int8_t>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<uint8_t>(),
      paddle::DataType::UINT8,
      common::errors::InvalidArgument("TestDtype<uint8_t>() should be equal to "
                                      "paddle::DataType::UINT8, but got %s",
                                      TestDtype<uint8_t>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<int16_t>(),
      paddle::DataType::INT16,
      common::errors::InvalidArgument("TestDtype<int16_t>() should be equal to "
                                      "paddle::DataType::INT16, but got %s",
                                      TestDtype<int16_t>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<int32_t>(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("TestDtype<int32_t>() should be equal to "
                                      "paddle::DataType::INT32, but got %s",
                                      TestDtype<int32_t>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<int64_t>(),
      paddle::DataType::INT64,
      common::errors::InvalidArgument("TestDtype<int64_t>() should be equal to "
                                      "paddle::DataType::INT64, but got %s",
                                      TestDtype<int64_t>()));
  PADDLE_ENFORCE_EQ(TestDtype<paddle::float16>(),
                    paddle::DataType::FLOAT16,
                    common::errors::InvalidArgument(
                        "TestDtype<paddle::float16>() should be equal to "
                        "paddle::DataType::FLOAT16, but got %s",
                        TestDtype<paddle::float16>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<float>(),
      paddle::DataType::FLOAT32,
      common::errors::InvalidArgument("TestDtype<float>() should be equal to "
                                      "paddle::DataType::FLOAT32, but got %s",
                                      TestDtype<float>()));
  PADDLE_ENFORCE_EQ(
      TestDtype<double>(),
      paddle::DataType::FLOAT64,
      common::errors::InvalidArgument("TestDtype<double>() should be equal to "
                                      "paddle::DataType::FLOAT64, but got %s",
                                      TestDtype<double>()));
  PADDLE_ENFORCE_EQ(TestDtype<paddle::complex64>(),
                    paddle::DataType::COMPLEX64,
                    common::errors::InvalidArgument(
                        "TestDtype<paddle::complex64>() should be equal to "
                        "paddle::DataType::COMPLEX64, but got %s",
                        TestDtype<paddle::complex64>()));
  PADDLE_ENFORCE_EQ(TestDtype<paddle::complex128>(),
                    paddle::DataType::COMPLEX128,
                    common::errors::InvalidArgument(
                        "TestDtype<paddle::complex128>() should be equal to "
                        "paddle::DataType::COMPLEX128, but got %s",
                        TestDtype<paddle::complex128>()));
}

void TestInitialized() {
  auto test_tensor = paddle::experimental::empty({1, 1});
  PADDLE_ENFORCE_EQ(test_tensor.initialized(),
                    true,
                    common::errors::InvalidArgument(
                        "test_tensor should be initialized, but got %s",
                        test_tensor.initialized()));
  float* tensor_data = test_tensor.data<float>();
  for (int i = 0; i < test_tensor.size(); i++) {
    tensor_data[i] = 0.5;
  }
  for (int i = 0; i < test_tensor.size(); i++) {
    PADDLE_ENFORCE_EQ(tensor_data[i],
                      0.5,
                      common::errors::InvalidArgument(
                          "tensor_data[%d] should be equal to 0.5, "
                          "but got %f",
                          i,
                          tensor_data[i]));
  }
}

void TestDataInterface() {
  // Test DenseTensor
  auto test_tensor = paddle::experimental::empty({1, 1});
  PADDLE_ENFORCE_EQ(test_tensor.initialized(),
                    true,
                    common::errors::InvalidArgument(
                        "test_tensor should be initialized, but got %s",
                        test_tensor.initialized()));
  void* tensor_ptr = test_tensor.data();
  PADDLE_ENFORCE_NE(
      tensor_ptr,
      nullptr,
      common::errors::InvalidArgument(
          "test_tensor should not be NULL, but got %p", tensor_ptr));
  const void* const_tensor_ptr = test_tensor.data();
  PADDLE_ENFORCE_NE(
      const_tensor_ptr,
      nullptr,
      common::errors::InvalidArgument("const_tensor should not be NULL, "
                                      "but got %p",
                                      const_tensor_ptr));
  // Test SelectedRows
  std::vector<int64_t> rows = {0};
  std::shared_ptr<phi::SelectedRows> selected_rows =
      std::make_shared<phi::SelectedRows>(rows, 1);
  selected_rows->mutable_value()->Resize(common::make_ddim({1, 1}));
  selected_rows->mutable_value()->mutable_data<float>(phi::CPUPlace())[0] =
      static_cast<float>(10.0f);
  paddle::Tensor sr_tensor = paddle::Tensor(selected_rows);
  PADDLE_ENFORCE_EQ(sr_tensor.initialized(),
                    true,
                    common::errors::InvalidArgument(
                        "sr_tensor should be initialized, but got %s",
                        sr_tensor.initialized()));
  tensor_ptr = sr_tensor.data();
  PADDLE_ENFORCE_NE(tensor_ptr,
                    nullptr,
                    common::errors::InvalidArgument(
                        "tensor should not be NULL, but got %p", tensor_ptr));
  const_tensor_ptr = sr_tensor.data();
  PADDLE_ENFORCE_NE(
      const_tensor_ptr,
      nullptr,
      common::errors::InvalidArgument("const_tensor should not be NULL, "
                                      "but got %p",
                                      const_tensor_ptr));
}

void TestJudgeTensorType() {
  Tensor test_tensor(phi::CPUPlace(), {1, 1});
  PADDLE_ENFORCE_EQ(
      test_tensor.is_dense_tensor(),
      true,
      common::errors::InvalidArgument("test_tensor should be a dense tensor, "
                                      "but got %s",
                                      test_tensor.is_dense_tensor()));
}

TEST(PhiTensor, All) {
  VLOG(2) << "TestCopy";
  GroupTestCopy();
  VLOG(2) << "TestDtype";
  GroupTestDtype();
  VLOG(2) << "TestShape";
  TestAPISizeAndShape();
  VLOG(2) << "TestPlace";
  TestAPIPlace();
  VLOG(2) << "TestSlice";
  TestAPISlice();
  VLOG(2) << "TestCast";
  GroupTestCast();
  VLOG(2) << "TestInitialized";
  TestInitialized();
  VLOG(2) << "TestDataInterface";
  TestDataInterface();
  VLOG(2) << "TestJudgeTensorType";
  TestJudgeTensorType();
}

}  // namespace tests
}  // namespace paddle
