// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/framework/data_type_transform.h"
#include "paddle/phi/core/kernel_factory.h"

template <typename InT, typename OutT>
void TransformTest(const phi::KernelKey& kernel_type_for_var,
                   const phi::KernelKey& expected_kernel_type,
                   const phi::CPUPlace& cpu_place,
                   const phi::XPUPlace& xpu_place,
                   const InT* cpu_data,
                   const int data_number) {
  phi::XPUContext context(xpu_place);
  phi::DenseTensor in;
  phi::DenseTensor in_xpu;
  phi::DenseTensor out;
  phi::DenseTensor out_xpu;

  // copy from cpu_data to cpu tensor
  InT* in_ptr =
      in.mutable_data<InT>(common::make_ddim({data_number}), cpu_place);
  memcpy(in_ptr, cpu_data, sizeof(InT) * data_number);

  // test case 1: on xpu
  {
    // copy from cpu tensor to xpu tensor
    paddle::framework::TensorCopy(in, xpu_place, context, &in_xpu);
    context.Wait();

    // call trans data
    phi::TransDataType(
        kernel_type_for_var, expected_kernel_type, in_xpu, &out_xpu);

    // copy from xpu tensor to cpu tensor
    paddle::framework::TensorCopy(out_xpu, cpu_place, context, &out);
    context.Wait();

    // check result
    OutT* out_ptr = out.data<OutT>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_ptr[i], static_cast<OutT>(cpu_data[i]));
    }
  }

  // test case 2: on cpu
  {
    // call trans data
    phi::TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);

    // check result
    OutT* out_ptr = out.data<OutT>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_ptr[i], static_cast<OutT>(cpu_data[i]));
    }
  }
}

TEST(DataTypeTransform, XPUTransform) {
  auto cpu_place = phi::CPUPlace();
  auto xpu_place = phi::XPUPlace(0);
  phi::XPUContext context(xpu_place);

  auto kernel_fp16 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT16);
  auto kernel_fp32 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);
  auto kernel_fp64 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT64);
  auto kernel_int16 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT16);
  auto kernel_int32 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT32);
  auto kernel_int64 = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT64);
  auto kernel_bool = phi::KernelKey(
      xpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::BOOL);

  {
    // float16 -> any
    phi::dtype::float16 cpu_data[6] = {phi::dtype::float16(0),
                                       phi::dtype::float16(1),
                                       phi::dtype::float16(2),
                                       phi::dtype::float16(3),
                                       phi::dtype::float16(4),
                                       phi::dtype::float16(5)};
    TransformTest<phi::dtype::float16, float>(
        kernel_fp16, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<phi::dtype::float16, double>(
        kernel_fp16, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<phi::dtype::float16, int32_t>(
        kernel_fp16, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<phi::dtype::float16, int64_t>(
        kernel_fp16, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<phi::dtype::float16, bool>(
        kernel_fp16, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // float -> any
    float cpu_data[6] = {0, 1, 2, 3, 4, 5};
    TransformTest<float, phi::dtype::float16>(
        kernel_fp32, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, float>(
        kernel_fp32, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, double>(
        kernel_fp32, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, int16_t>(
        kernel_fp32, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, int32_t>(
        kernel_fp32, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, int64_t>(
        kernel_fp32, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<float, bool>(
        kernel_fp32, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // double -> any
    double cpu_data[6] = {0, 1, 2, 3, 4, 5};
    TransformTest<double, phi::dtype::float16>(
        kernel_fp64, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, float>(
        kernel_fp64, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, double>(
        kernel_fp64, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, int16_t>(
        kernel_fp64, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, int32_t>(
        kernel_fp64, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, int64_t>(
        kernel_fp64, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<double, bool>(
        kernel_fp64, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // int16 -> any
    int16_t cpu_data[6] = {0, 1, 2, 3, 4, 5};
    TransformTest<int16_t, phi::dtype::float16>(
        kernel_int16, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, float>(
        kernel_int16, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, double>(
        kernel_int16, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, int16_t>(
        kernel_int16, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, int32_t>(
        kernel_int16, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, int64_t>(
        kernel_int16, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int16_t, bool>(
        kernel_int16, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // int32 -> any
    int32_t cpu_data[6] = {0, 1, 2, 3, 4, 5};
    TransformTest<int32_t, phi::dtype::float16>(
        kernel_int32, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, float>(
        kernel_int32, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, double>(
        kernel_int32, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, int16_t>(
        kernel_int32, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, int32_t>(
        kernel_int32, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, int64_t>(
        kernel_int32, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int32_t, bool>(
        kernel_int32, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // int64 -> any
    int64_t cpu_data[6] = {0, 1, 2, 3, 4, 5};
    TransformTest<int64_t, phi::dtype::float16>(
        kernel_int64, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, float>(
        kernel_int64, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, double>(
        kernel_int64, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, int16_t>(
        kernel_int64, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, int32_t>(
        kernel_int64, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, int64_t>(
        kernel_int64, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<int64_t, bool>(
        kernel_int64, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
  {
    // bool -> any
    bool cpu_data[6] = {0, 1, 0, 1, 1, 0};
    TransformTest<bool, phi::dtype::float16>(
        kernel_bool, kernel_fp16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, float>(
        kernel_bool, kernel_fp32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, double>(
        kernel_bool, kernel_fp64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, int16_t>(
        kernel_bool, kernel_int16, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, int32_t>(
        kernel_bool, kernel_int32, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, int64_t>(
        kernel_bool, kernel_int64, cpu_place, xpu_place, cpu_data, 6);
    TransformTest<bool, bool>(
        kernel_bool, kernel_bool, cpu_place, xpu_place, cpu_data, 6);
  }
}
