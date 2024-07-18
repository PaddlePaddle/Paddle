/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_type_transform.h"

#include "gtest/gtest.h"

TEST(DataTypeTransform, CPUTransform) {
  auto place = phi::CPUPlace();

  auto kernel_fp16 = phi::KernelKey(
      place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT16);

  auto kernel_bf16 = phi::KernelKey(
      place, phi::DataLayout::ALL_LAYOUT, phi::DataType::BFLOAT16);

  auto kernel_fp32 = phi::KernelKey(
      place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);

  auto kernel_fp64 = phi::KernelKey(
      place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT64);

  auto kernel_int32 =
      phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT32);

  auto kernel_int64 =
      phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT64);

  auto kernel_bool =
      phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, phi::DataType::BOOL);

  // data type transform from float32
  {
    phi::DenseTensor in;
    phi::DenseTensor out;

    float* ptr = in.mutable_data<float>(common::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i / 3;  // NOLINT
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(i / 3));  // NOLINT
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(i / 3));
    }
  }

  // data type transform from/to float16
  {
    phi::DenseTensor in;
    phi::DenseTensor out;

    phi::dtype::float16* ptr =
        in.mutable_data<phi::dtype::float16>(common::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from float16 to other data types
    paddle::framework::TransDataType(kernel_fp16, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to float16
    float* in_data_float =
        in.mutable_data<float>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = static_cast<float>(i);
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_fp16, in, &out);
    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_float[i]).x);
    }

    // transform double to float16
    double* in_data_double =
        in.mutable_data<double>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_fp16, in, &out);
    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<phi::dtype::float16>(in_data_double[i]).x);
    }

    // transform int to float16
    int* in_data_int = in.mutable_data<int>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    paddle::framework::TransDataType(kernel_int32, kernel_fp16, in, &out);
    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_int[i]).x);
    }

    // transform int64 to float16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_fp16, in, &out);
    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_int64[i]).x);
    }

    // transform bool to float16
    bool* in_data_bool =
        in.mutable_data<bool>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_fp16, in, &out);
    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_bool[i]).x);
    }
  }

  // data type transform from/to bfloat16
  {
    phi::DenseTensor in;
    phi::DenseTensor out;

    phi::dtype::bfloat16* ptr =
        in.mutable_data<phi::dtype::bfloat16>(common::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from bfloat16 to other data types
    paddle::framework::TransDataType(kernel_bf16, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to bfloat16
    float* in_data_float =
        in.mutable_data<float>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = static_cast<float>(i);
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_bf16, in, &out);
    ptr = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<phi::dtype::bfloat16>(in_data_float[i]).x);
    }

    // transform double to bfloat16
    double* in_data_double =
        in.mutable_data<double>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_bf16, in, &out);
    ptr = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<phi::dtype::bfloat16>(in_data_double[i]).x);
    }

    // transform int to bfloat16
    int* in_data_int = in.mutable_data<int>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bf16, in, &out);
    ptr = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::bfloat16>(in_data_int[i]).x);
    }

    // transform int64 to bfloat16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_bf16, in, &out);
    ptr = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<phi::dtype::bfloat16>(in_data_int64[i]).x);
    }

    // transform bool to bfloat16
    bool* in_data_bool =
        in.mutable_data<bool>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_bf16, in, &out);
    ptr = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::bfloat16>(in_data_bool[i]).x);
    }
  }

  // data type transform from/to int32
  {
    phi::DenseTensor in;
    phi::DenseTensor out;

    int32_t* ptr = in.mutable_data<int32_t>(common::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from int32 to other data types
    paddle::framework::TransDataType(kernel_int32, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bf16, in, &out);
    phi::dtype::bfloat16* out_data_bf16 = out.data<phi::dtype::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bf16[i], static_cast<phi::dtype::bfloat16>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to int32
    float* in_data_float =
        in.mutable_data<float>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = static_cast<float>(i);
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_float[i]));
    }

    // transform double to int32
    double* in_data_double =
        in.mutable_data<double>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_double[i]));
    }

    // transform bfloat16 to int32
    phi::dtype::bfloat16* in_data_bf16 =
        in.mutable_data<phi::dtype::bfloat16>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bf16[i] = i;
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_bf16[i]));
    }

    // transform int64 to int32
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_int64[i]));
    }

    // transform bool to int32
    bool* in_data_bool =
        in.mutable_data<bool>(common::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_bool[i]));
    }
  }
}
